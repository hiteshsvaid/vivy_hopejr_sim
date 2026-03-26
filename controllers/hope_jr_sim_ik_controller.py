#!/usr/bin/env python3

import argparse
import importlib.util
import json
import socket
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ui.hope_jr_teleop_status_ui import HopeJrTeleopStatusUi
from controllers.quest_teleop_mapper import QuestTeleopMapper

import numpy as np
from scipy.spatial.transform import Rotation


DEFAULT_LEROBOT_REPO = Path("/home/viaan/huggingface/lerobot")
DEFAULT_PACKET_PATH = Path("/tmp/hope_jr_quest_latest.json")
DEFAULT_IK_SPEC_PATH = DEFAULT_LEROBOT_REPO / "src/lerobot/robots/hope_jr/hope_jr_arm_ik_spec.json"
DEFAULT_KINEMATICS_MODULE_PATH = DEFAULT_LEROBOT_REPO / "src/lerobot/robots/hope_jr/hope_jr_arm_kinematics.py"
DEFAULT_ARTICULATION_ROOT_PATH = "/World/JointTest"
DEFAULT_JOINT_ROOT_PATH = "/World/JointTest/Joints"
DEFAULT_UDP_LISTEN_HOST = "127.0.0.1"
DEFAULT_UDP_LISTEN_PORT = 8766
DEFAULT_DEBUG_PATH = Path("/tmp/hope_jr_sim_ik_debug.json")
DEFAULT_TELEOP_DEBUG_ROOT = "/World/JointTest/TeleopDebug"
DEFAULT_END_EFFECTOR_PATH = "/World/JointTest/PalmBody/EndEffector"
DEFAULT_EVENT_LOG_PATH = Path("/tmp/hope_jr_sim_ik_events.ndjson")
DEFAULT_PACKET_STALE_TIMEOUT_S = 0.75
DEFAULT_STOP_TARGETS_DEG = {
    "right_elbow": 40.5,
}
DEFAULT_MODEL_JOINT_SIGNS = {
    "right_shoulder_pitch": -1.0,
    "right_shoulder_yaw": -1.0,
    "right_upper_elbow": -1.0,
    "right_elbow": -1.0,
    "right_forearm_twist": 1.0,
    "right_wrist": 1.0,
    "right_palm": 1.0,
}

_ACTIVE_LOOP = None


class HopeJrSimIkController:
    def __init__(
        self,
        *,
        lerobot_repo: Path,
        packet_path: Path,
        joint_root_path: str,
        position_scale: float,
        world_offset: np.ndarray,
        world_rotate_xyz_deg: np.ndarray,
        quest_position_axes: tuple[int, int, int],
        quest_position_signs: np.ndarray,
        position_only: bool,
        debug_path: Path,
        use_udp: bool,
        udp_listen_host: str,
        udp_listen_port: int,
        teleop_debug_root: str,
        show_teleop_debug: bool,
        anchor_delay_s: float,
        grip_threshold: float,
        event_log_path: Path,
        quest_deadband_m: float,
        packet_stale_timeout_s: float,
        end_effector_path: str,
        write_joint_state_directly: bool,
        active_joint_names: tuple[str, ...] | None = None,
        inactive_joint_behavior: str = "neutral",
    ):
        self.lerobot_repo = lerobot_repo
        self.packet_path = packet_path
        self.joint_root_path = joint_root_path.rstrip("/")
        self.articulation_root_path = self.joint_root_path.rsplit("/", 1)[0]
        self.position_only = position_only
        self.debug_path = debug_path
        self.use_udp = use_udp
        self.udp_listen_host = udp_listen_host
        self.udp_listen_port = udp_listen_port
        self.teleop_debug_root = teleop_debug_root.rstrip("/")
        self.show_teleop_debug = show_teleop_debug
        self.anchor_delay_s = anchor_delay_s
        self.event_log_path = event_log_path
        self.packet_stale_timeout_s = float(packet_stale_timeout_s)
        self.end_effector_path = end_effector_path
        self.write_joint_state_directly = bool(write_joint_state_directly)
        if active_joint_names is None:
            self.active_joint_names = tuple()
        else:
            self.active_joint_names = tuple(active_joint_names)
        if inactive_joint_behavior not in {"neutral", "hold"}:
            raise ValueError(f"Unsupported inactive_joint_behavior: {inactive_joint_behavior}")
        self.inactive_joint_behavior = inactive_joint_behavior
        self._udp_socket = None
        if self.use_udp:
            self._udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._udp_socket.bind((self.udp_listen_host, self.udp_listen_port))
            self._udp_socket.setblocking(False)
        self.kinematics_module = self._load_kinematics_module(
            DEFAULT_KINEMATICS_MODULE_PATH
            if lerobot_repo == DEFAULT_LEROBOT_REPO
            else lerobot_repo / "src/lerobot/robots/hope_jr/hope_jr_arm_kinematics.py"
        )
        self.model = self.kinematics_module.HopeJrArmKinematics.from_json(
            DEFAULT_IK_SPEC_PATH
            if lerobot_repo == DEFAULT_LEROBOT_REPO
            else lerobot_repo / "src/lerobot/robots/hope_jr/hope_jr_arm_ik_spec.json"
        )
        self.model_joint_signs = np.asarray(
            [DEFAULT_MODEL_JOINT_SIGNS.get(name, 1.0) for name in self.model.joint_names],
            dtype=float,
        )
        self.last_joint_targets_deg = np.zeros(len(self.model.joint_names), dtype=float)
        self.neutral_model_joint_targets_deg = np.array([DEFAULT_STOP_TARGETS_DEG.get(name, 0.0) for name in self.model.joint_names], dtype=float)
        self.inactive_joint_hold_targets_deg = self.neutral_model_joint_targets_deg.copy()
        if self.active_joint_names:
            active_set = set(self.active_joint_names)
            self.active_joint_mask = np.array([1.0 if name in active_set else 0.0 for name in self.model.joint_names], dtype=float)
        else:
            self.active_joint_mask = np.ones(len(self.model.joint_names), dtype=float)
        self.last_packet_timestamp = None
        self.minimum_packet_timestamp = None
        self.last_debug_payload = None
        self._last_event_key = None
        self.teleop_mapper = QuestTeleopMapper(
            position_scale=position_scale,
            world_offset=world_offset,
            world_rotate_xyz_deg=world_rotate_xyz_deg,
            quest_position_axes=quest_position_axes,
            quest_position_signs=quest_position_signs,
            position_only=position_only,
            anchor_delay_s=anchor_delay_s,
            grip_threshold=grip_threshold,
            quest_deadband_m=quest_deadband_m,
            make_pose=self.kinematics_module.make_pose,
        )
        self._a_pressed_last = False
        self.last_hand_state = {}
        self.last_packet_received_at = None
        self._articulation = None
        self._articulation_joint_indices = None

    def _inactive_joint_reference_targets_deg(self) -> np.ndarray:
        if self.inactive_joint_behavior == "hold":
            return self.inactive_joint_hold_targets_deg
        return self.neutral_model_joint_targets_deg

    def _load_kinematics_module(self, module_path: Path):
        spec = importlib.util.spec_from_file_location("hope_jr_arm_kinematics", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load Hope Jr kinematics module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def _load_latest_packet(self) -> dict[str, Any] | None:
        if self._udp_socket is not None:
            latest_payload = None
            while True:
                try:
                    payload, _addr = self._udp_socket.recvfrom(1024 * 1024)
                except BlockingIOError:
                    break
                latest_payload = payload
            if latest_payload is None:
                return None
            return json.loads(latest_payload.decode("utf-8"))
        if not self.packet_path.is_file():
            return None
        return json.loads(self.packet_path.read_text())

    def _packet_to_target_pose(
        self,
        packet: dict[str, Any],
        current_joint_targets_deg: np.ndarray,
    ) -> Any | None:
        anchor_joint_targets_deg = current_joint_targets_deg.copy()
        if self.active_joint_names:
            inactive = self.active_joint_mask < 0.5
            if self.inactive_joint_behavior == "hold":
                self.inactive_joint_hold_targets_deg = current_joint_targets_deg.copy()
            inactive_reference_targets_deg = self._inactive_joint_reference_targets_deg()
            anchor_joint_targets_deg[inactive] = inactive_reference_targets_deg[inactive]
        current_sim_pose = self.model.forward_kinematics(anchor_joint_targets_deg)
        stage = self._get_stage()
        current_stage_pose = self._read_stage_end_effector_pose(stage)
        map_result = self.teleop_mapper.map_packet(
            packet,
            current_sim_pose=current_sim_pose,
            current_stage_pose=current_stage_pose,
            active_joint_names=self.active_joint_names,
            inactive_joint_behavior=self.inactive_joint_behavior,
            anchor_joint_targets_deg=anchor_joint_targets_deg,
        )
        if map_result is None:
            return None
        if map_result.waiting_for_anchor:
            self._update_teleop_debug_visuals(
                quest_anchor_position=map_result.quest_current_position,
                quest_current_position=map_result.quest_current_position,
                quest_mapped_position=map_result.quest_mapped_position_stage,
                sim_target_position=map_result.sim_target_position_stage,
                actual_end_effector_position=self._read_stage_end_effector_position(stage),
                actual_end_effector_pose=self._read_stage_end_effector_pose(stage),
                waiting_for_anchor=True,
            )
            return map_result
        if map_result.anchor_captured_payload is not None:
            self._append_event(
                map_result.anchor_captured_payload,
                dedupe_key=("anchor_captured", tuple(np.round(self.teleop_mapper.quest_anchor_position, 6))),
            )
        self._update_teleop_debug_visuals(
            quest_anchor_position=map_result.quest_anchor_position,
            quest_current_position=map_result.quest_current_position,
            quest_mapped_position=map_result.quest_mapped_position_stage,
            sim_target_position=map_result.sim_target_position_stage,
            actual_end_effector_position=self._read_stage_end_effector_position(stage),
            actual_end_effector_pose=self._read_stage_end_effector_pose(stage),
        )
        return map_result


    def _append_event(self, payload: dict[str, Any], *, dedupe_key: tuple[Any, ...] | None = None) -> None:
        if dedupe_key is not None and dedupe_key == self._last_event_key:
            return
        self._last_event_key = dedupe_key
        event = {"logged_at": time.time(), **payload}
        with self.event_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event) + "\n")

    def _write_debug(self, payload: dict[str, Any]) -> None:
        self.last_debug_payload = payload
        tmp_path = self.debug_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2) + "\n")
        tmp_path.replace(self.debug_path)

    def _update_teleop_debug_visuals(
        self,
        *,
        quest_anchor_position: np.ndarray,
        quest_current_position: np.ndarray,
        quest_mapped_position: np.ndarray,
        sim_target_position: np.ndarray,
        actual_end_effector_position: np.ndarray | None = None,
        actual_end_effector_pose: np.ndarray | None = None,
        waiting_for_anchor: bool = False,
    ) -> None:
        if not self.show_teleop_debug:
            return
        stage = self._get_stage()
        if stage is None:
            return
        try:
            from pxr import Gf, Sdf, UsdGeom
        except ImportError:
            return

        root = stage.DefinePrim(self.teleop_debug_root, "Xform")
        sim_target_color = (0.0, 1.0, 0.0) if waiting_for_anchor else (1.0, 0.0, 0.0)
        visuals = [
            ("QuestMapped", quest_mapped_position, (1.0, 0.5, 0.0), 0.004),
            ("SimTarget", sim_target_position, sim_target_color, 0.005),
        ]
        for name, position, color, radius in visuals:
            sphere_path = f"{self.teleop_debug_root}/{name}"
            sphere = UsdGeom.Sphere.Define(stage, sphere_path)
            prim = sphere.GetPrim()
            display_attr = prim.GetAttribute("primvars:displayColor")
            if not display_attr.IsValid():
                display_attr = prim.CreateAttribute("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray)
            display_attr.Set([Gf.Vec3f(*color)])
            radius_attr = sphere.GetRadiusAttr()
            radius_attr.Set(radius)
            translate_attr = prim.GetAttribute("xformOp:translate")
            if not translate_attr.IsValid():
                translate_attr = prim.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3)
            translate_attr.Set(Gf.Vec3d(*[float(v) for v in position]))
            order_attr = prim.GetAttribute("xformOpOrder")
            if not order_attr.IsValid() or not order_attr.Get():
                order_attr.Set(["xformOp:translate"])

        if actual_end_effector_pose is not None:
            arrow_root_path = f"{self.teleop_debug_root}/ActualEndEffectorArrow"
            arrow_root = stage.DefinePrim(arrow_root_path, "Xform")
            arrow_rotation = actual_end_effector_pose[:3, :3]
            quat_xyzw = Rotation.from_matrix(arrow_rotation).as_quat()
            quat_wxyz = [float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])]
            translate_attr = arrow_root.GetAttribute("xformOp:translate")
            if not translate_attr.IsValid():
                translate_attr = arrow_root.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3)
            translate_attr.Set(Gf.Vec3d(*[float(v) for v in actual_end_effector_pose[:3, 3]]))
            orient_attr = arrow_root.GetAttribute("xformOp:orient")
            if not orient_attr.IsValid():
                orient_attr = arrow_root.CreateAttribute("xformOp:orient", Sdf.ValueTypeNames.Quatf)
            orient_attr.Set(Gf.Quatf(quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]))
            order_attr = arrow_root.GetAttribute("xformOpOrder")
            order_attr.Set(["xformOp:translate", "xformOp:orient"])

            shaft = UsdGeom.Cylinder.Define(stage, f"{arrow_root_path}/Shaft")
            shaft_prim = shaft.GetPrim()
            shaft.GetRadiusAttr().Set(0.0028)
            shaft.GetHeightAttr().Set(0.03)
            shaft_display = shaft_prim.GetAttribute("primvars:displayColor")
            if not shaft_display.IsValid():
                shaft_display = shaft_prim.CreateAttribute("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray)
            shaft_display.Set([Gf.Vec3f(0.1, 0.5, 1.0)])
            shaft_translate = shaft_prim.GetAttribute("xformOp:translate")
            if not shaft_translate.IsValid():
                shaft_translate = shaft_prim.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3)
            shaft_translate.Set(Gf.Vec3d(0.0, 0.0, 0.015))
            shaft_order = shaft_prim.GetAttribute("xformOpOrder")
            shaft_order.Set(["xformOp:translate"])

            tip = UsdGeom.Cone.Define(stage, f"{arrow_root_path}/Tip")
            tip_prim = tip.GetPrim()
            tip.GetRadiusAttr().Set(0.005)
            tip.GetHeightAttr().Set(0.014)
            tip_display = tip_prim.GetAttribute("primvars:displayColor")
            if not tip_display.IsValid():
                tip_display = tip_prim.CreateAttribute("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray)
            tip_display.Set([Gf.Vec3f(0.1, 0.5, 1.0)])
            tip_translate = tip_prim.GetAttribute("xformOp:translate")
            if not tip_translate.IsValid():
                tip_translate = tip_prim.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3)
            tip_translate.Set(Gf.Vec3d(0.0, 0.0, 0.032))
            tip_order = tip_prim.GetAttribute("xformOpOrder")
            tip_order.Set(["xformOp:translate"])

    def _read_stage_end_effector_pose(self, stage) -> np.ndarray | None:
        if stage is None:
            return None
        try:
            from pxr import UsdGeom
        except ImportError:
            return None
        prim = stage.GetPrimAtPath(self.end_effector_path)
        if not prim.IsValid():
            return None
        try:
            xform_cache = UsdGeom.XformCache()
            world_transform = xform_cache.GetLocalToWorldTransform(prim)
            matrix = np.array(world_transform, dtype=float).T
            return matrix
        except Exception:
            return None

    def _read_stage_end_effector_position(self, stage) -> np.ndarray | None:
        pose = self._read_stage_end_effector_pose(stage)
        if pose is None:
            return None
        return pose[:3, 3].copy()

    def _get_stage(self):
        try:
            import omni.usd
        except ImportError:
            return None
        return omni.usd.get_context().get_stage()

    def _get_articulation(self):
        try:
            from isaacsim.core.prims import SingleArticulation
            from isaacsim.core.utils.types import ArticulationAction
        except ImportError:
            return None
        if self._articulation is None:
            self._articulation = SingleArticulation(self.articulation_root_path, reset_xform_properties=False)
            self._articulation_joint_indices = None
        try:
            if not self._articulation.handles_initialized:
                self._articulation.initialize()
        except Exception:
            return None
        if self._articulation_joint_indices is None:
            try:
                self._articulation_joint_indices = np.array(
                    [self._articulation.get_dof_index(name) for name in self.model.joint_names],
                    dtype=np.int64,
                )
            except Exception:
                return None
        return self._articulation

    def _get_articulation_joint_indices(self) -> np.ndarray | None:
        articulation = self._get_articulation()
        if articulation is None:
            return None
        return self._articulation_joint_indices

    def _stage_to_model_joint_positions_deg(self, joint_positions_deg: np.ndarray) -> np.ndarray:
        return np.asarray(joint_positions_deg, dtype=float) * self.model_joint_signs

    def _model_to_stage_joint_positions_deg(self, joint_positions_deg: np.ndarray) -> np.ndarray:
        return np.asarray(joint_positions_deg, dtype=float) * self.model_joint_signs

    def _read_current_joint_targets_deg(self, stage) -> np.ndarray:
        articulation = self._get_articulation()
        joint_indices = self._get_articulation_joint_indices()
        if articulation is not None and joint_indices is not None:
            try:
                joint_positions_rad = articulation.get_joint_positions(joint_indices=joint_indices)
                if joint_positions_rad is not None:
                    return np.rad2deg(np.asarray(joint_positions_rad, dtype=float))
            except Exception:
                pass
        joint_targets = []
        for joint_name in self.model.joint_names:
            prim = stage.GetPrimAtPath(f"{self.joint_root_path}/{joint_name}")
            if not prim.IsValid():
                raise RuntimeError(f"Joint prim not found: {self.joint_root_path}/{joint_name}")
            state_attr = prim.GetAttribute("state:angular:physics:position")
            state_value = state_attr.Get() if state_attr.IsValid() else None
            if state_value is not None:
                joint_targets.append(float(state_value))
                continue
            attr = prim.GetAttribute("drive:angular:physics:targetPosition")
            value = attr.Get() if attr.IsValid() else None
            joint_targets.append(float(value) if value is not None else 0.0)
        return np.asarray(joint_targets, dtype=float)

    def _read_stage_joint_positions_deg(self, stage) -> np.ndarray | None:
        articulation = self._get_articulation()
        joint_indices = self._get_articulation_joint_indices()
        if articulation is not None and joint_indices is not None:
            try:
                joint_positions_rad = articulation.get_joint_positions(joint_indices=joint_indices)
                if joint_positions_rad is not None:
                    return np.rad2deg(np.asarray(joint_positions_rad, dtype=float))
            except Exception:
                pass
        if stage is None:
            return None
        joint_positions = []
        for joint_name in self.model.joint_names:
            prim = stage.GetPrimAtPath(f"{self.joint_root_path}/{joint_name}")
            if not prim.IsValid():
                return None
            state_attr = prim.GetAttribute("state:angular:physics:position")
            state_value = state_attr.Get() if state_attr.IsValid() else None
            if state_value is None:
                return None
            joint_positions.append(float(state_value))
        return np.asarray(joint_positions, dtype=float)

    def _write_joint_targets_deg(self, stage, joint_targets_deg: np.ndarray) -> None:
        articulation = self._get_articulation()
        joint_indices = self._get_articulation_joint_indices()
        if articulation is not None and joint_indices is not None:
            try:
                from isaacsim.core.utils.types import ArticulationAction
                articulation.apply_action(
                    ArticulationAction(
                        joint_positions=np.deg2rad(np.asarray(joint_targets_deg, dtype=float)),
                        joint_indices=joint_indices,
                    )
                )
                return
            except Exception:
                pass
        for joint_name, target_deg in zip(self.model.joint_names, joint_targets_deg, strict=True):
            prim = stage.GetPrimAtPath(f"{self.joint_root_path}/{joint_name}")
            if not prim.IsValid():
                raise RuntimeError(f"Joint prim not found: {self.joint_root_path}/{joint_name}")
            attr = prim.GetAttribute("drive:angular:physics:targetPosition")
            if not attr.IsValid():
                raise RuntimeError(
                    f"Joint drive target attribute missing on {self.joint_root_path}/{joint_name}"
                )
            attr.Set(float(target_deg))


    def _write_joint_state_deg(self, stage, joint_positions_deg: np.ndarray) -> None:
        articulation = self._get_articulation()
        joint_indices = self._get_articulation_joint_indices()
        if articulation is not None and joint_indices is not None:
            try:
                articulation.set_joint_positions(np.deg2rad(np.asarray(joint_positions_deg, dtype=float)), joint_indices=joint_indices)
                return
            except Exception:
                pass
        for joint_name, position_deg in zip(self.model.joint_names, joint_positions_deg, strict=True):
            prim = stage.GetPrimAtPath(f"{self.joint_root_path}/{joint_name}")
            if not prim.IsValid():
                continue
            pos_attr = prim.GetAttribute("state:angular:physics:position")
            vel_attr = prim.GetAttribute("state:angular:physics:velocity")
            if pos_attr.IsValid():
                pos_attr.Set(float(position_deg))
            if vel_attr.IsValid():
                vel_attr.Set(0.0)

    def reset_target_positions(self, target_value_deg: float = 0.0, reset_joint_state: bool = True) -> None:
        self.teleop_mapper.reset()
        stage = self._get_stage()
        if stage is None:
            return
        target_values = np.array(
            [DEFAULT_STOP_TARGETS_DEG.get(joint_name, float(target_value_deg)) for joint_name in self.model.joint_names],
            dtype=float,
        )
        self._write_joint_targets_deg(stage, target_values)
        if reset_joint_state:
            self._write_joint_state_deg(stage, target_values)
        self.last_joint_targets_deg = self._stage_to_model_joint_positions_deg(target_values)

    def solve_once(self, *, apply_to_stage: bool) -> dict[str, Any] | None:
        packet = self._load_latest_packet()
        if packet is None:
            return None

        packet_timestamp = packet.get("timestamp")
        normalized = packet.get("normalized") if isinstance(packet, dict) else None
        hand = normalized.get("right_hand") if isinstance(normalized, dict) else None
        if isinstance(hand, dict):
            self.last_hand_state = dict(hand)
        a_pressed = bool(hand.get("a_pressed", False)) if isinstance(hand, dict) else False
        if a_pressed and not self._a_pressed_last:
            if apply_to_stage:
                self.reset_target_positions(reset_joint_state=True)
            reset_event = {
                "status": "reset_to_neutral",
                "packet_timestamp": packet_timestamp,
                "reason": "a_pressed",
            }
            self._append_event(reset_event, dedupe_key=("reset_to_neutral", packet_timestamp))
            self._write_debug(reset_event)
            self._a_pressed_last = True
            return {
                "timestamp": packet_timestamp,
                "status": "reset_to_neutral",
            }
        self._a_pressed_last = a_pressed
        self.last_packet_received_at = time.time()

        if packet_timestamp is None:
            if self.last_debug_payload is None:
                self._write_debug({"status": "ignored", "reason": "missing_timestamp", "packet": packet})
            return None
        if self.minimum_packet_timestamp is not None and packet_timestamp <= self.minimum_packet_timestamp:
            if self.last_debug_payload is None or self.last_debug_payload.get("status") != "applied":
                self._write_debug(
                    {
                        "status": "ignored",
                        "reason": "stale_timestamp",
                        "packet_timestamp": packet_timestamp,
                        "minimum_packet_timestamp": self.minimum_packet_timestamp,
                        "packet": packet,
                    }
                )
            return None
        if packet_timestamp == self.last_packet_timestamp:
            return None
        self.last_packet_timestamp = packet_timestamp

        stage = self._get_stage() if apply_to_stage else None
        if stage is not None:
            current_stage_joint_targets_deg = self._read_current_joint_targets_deg(stage)
            current_joint_targets_deg = self._stage_to_model_joint_positions_deg(current_stage_joint_targets_deg)
        else:
            current_stage_joint_targets_deg = None
            current_joint_targets_deg = self.last_joint_targets_deg

        map_result = self._packet_to_target_pose(packet, current_joint_targets_deg)
        if map_result is None:
            if self.teleop_mapper.quest_anchor_position is None and time.time() < self.teleop_mapper.anchor_ready_time:
                self._write_debug(
                    {
                        "status": "waiting_for_anchor",
                        "anchor_delay_s": self.anchor_delay_s,
                        "anchor_ready_time": self.teleop_mapper.anchor_ready_time,
                        "now": time.time(),
                    }
                )
            else:
                normalized = packet.get("normalized") if isinstance(packet, dict) else None
                hand = normalized.get("right_hand") if isinstance(normalized, dict) else None
                grip = hand.get("grip") if isinstance(hand, dict) else None
                ignored_event = {
                    "status": "ignored",
                    "reason": "packet_not_usable",
                    "grip": grip,
                    "grip_threshold": self.teleop_mapper.grip_threshold,
                }
                self._append_event(
                    ignored_event,
                    dedupe_key=("ignored", ignored_event["reason"], round(float(grip or 0.0), 3), round(float(self.teleop_mapper.grip_threshold), 3)),
                )
                if self.last_debug_payload is None or self.last_debug_payload.get("status") != "applied":
                    self._write_debug({**ignored_event, "packet": packet})
            return None
        if map_result.waiting_for_anchor:
            self._write_debug(
                {
                    "status": "waiting_for_anchor",
                    "anchor_delay_s": self.anchor_delay_s,
                    "anchor_ready_time": self.teleop_mapper.anchor_ready_time,
                    "now": time.time(),
                }
            )
            return None
        target_pose = map_result.target_pose
        hand_state = map_result.hand_state

        ik_seed_deg = current_joint_targets_deg.copy()
        if self.active_joint_names:
            inactive = self.active_joint_mask < 0.5
            inactive_reference_targets_deg = self._inactive_joint_reference_targets_deg()
            ik_seed_deg[inactive] = inactive_reference_targets_deg[inactive]
        solved_model_joint_targets_deg = self.model.inverse_kinematics(
            ik_seed_deg,
            target_pose,
            orientation_weight=0.0 if self.position_only else 1.0,
            active_joint_mask=self.active_joint_mask,
        )
        if self.active_joint_names:
            inactive = self.active_joint_mask < 0.5
            inactive_reference_targets_deg = self._inactive_joint_reference_targets_deg()
            solved_model_joint_targets_deg[inactive] = inactive_reference_targets_deg[inactive]
        self.last_joint_targets_deg = solved_model_joint_targets_deg
        solved_joint_targets_deg = self._model_to_stage_joint_positions_deg(solved_model_joint_targets_deg)

        stage_end_effector_position = self._read_stage_end_effector_position(stage) if stage is not None else None
        stage_joint_positions_deg = None
        stage_model_joint_positions_deg = None
        if stage is not None:
            self._write_joint_targets_deg(stage, solved_joint_targets_deg)
            if self.write_joint_state_directly:
                self._write_joint_state_deg(stage, solved_joint_targets_deg)
            stage_joint_positions_deg = self._read_stage_joint_positions_deg(stage)
            if stage_joint_positions_deg is not None:
                stage_model_joint_positions_deg = self._stage_to_model_joint_positions_deg(stage_joint_positions_deg)
            stage_end_effector_position = self._read_stage_end_effector_position(stage)

        achieved_pose = self.model.forward_kinematics(solved_model_joint_targets_deg)
        achieved_position = achieved_pose[:3, 3]
        target_position = target_pose[:3, 3]
        position_error = target_position - achieved_position
        target_stage_position = (self.teleop_mapper.model_to_stage_transform @ target_pose)[:3, 3]
        achieved_stage_position = (self.teleop_mapper.model_to_stage_transform @ achieved_pose)[:3, 3]
        stage_end_effector_error = None
        if stage_end_effector_position is not None:
            stage_end_effector_error = (target_stage_position - stage_end_effector_position).tolist()
        stage_vs_model_joint_delta = None
        if stage_model_joint_positions_deg is not None:
            stage_vs_model_joint_delta = (stage_model_joint_positions_deg - solved_model_joint_targets_deg).tolist()
        result = {
            "timestamp": packet_timestamp,
            "joint_names": self.model.joint_names,
            "joint_targets_deg": solved_joint_targets_deg.tolist(),
            "model_joint_targets_deg": solved_model_joint_targets_deg.tolist(),
            "stage_joint_positions_deg": None if stage_joint_positions_deg is None else stage_joint_positions_deg.tolist(),
            "stage_model_joint_positions_deg": None if stage_model_joint_positions_deg is None else stage_model_joint_positions_deg.tolist(),
            "stage_vs_model_joint_delta": stage_vs_model_joint_delta,
            "target_position": target_position.tolist(),
            "target_stage_position": target_stage_position.tolist(),
            "achieved_position": achieved_position.tolist(),
            "achieved_stage_position": achieved_stage_position.tolist(),
            "stage_end_effector_position": None if stage_end_effector_position is None else stage_end_effector_position.tolist(),
            "stage_end_effector_error": stage_end_effector_error,
            "position_error": position_error.tolist(),
            "position_only": self.position_only,
            "quest_position_axes": list(self.teleop_mapper.quest_position_axes),
            "quest_position_signs": self.teleop_mapper.quest_position_signs.tolist(),
            "grip": float(hand_state.get("grip", 0.0)),
            "trigger": float(hand_state.get("trigger", 0.0)),
            "active_joint_names": list(self.active_joint_names),
            "inactive_joint_behavior": self.inactive_joint_behavior,
        }
        hand_position = np.asarray(hand_state.get("position", [0.0, 0.0, 0.0]), dtype=float)
        event_payload = {
            "status": "applied" if stage is not None else "solved",
                "packet_timestamp": packet_timestamp,
                "position_only": self.position_only,
                "quest_position_axes": list(self.teleop_mapper.quest_position_axes),
                "quest_position_signs": self.teleop_mapper.quest_position_signs.tolist(),
                "raw_hand_position": hand_state.get("position"),
                "raw_hand_orientation_xyzw": hand_state.get("orientation_xyzw"),
                "quest_anchor_position": None if map_result.quest_anchor_position is None else map_result.quest_anchor_position.tolist(),
                "sim_anchor_position": None if self.teleop_mapper.sim_anchor_pose is None else self.teleop_mapper.sim_anchor_pose[:3, 3].tolist(),
                "quest_delta": None if map_result.quest_delta is None else map_result.quest_delta.tolist(),
                "mapped_delta": None if map_result.mapped_delta_model is None else map_result.mapped_delta_model.tolist(),
                "current_joint_targets_deg": current_joint_targets_deg.tolist(),
                "inactive_joint_behavior": self.inactive_joint_behavior,
                "result": result,
            }
        self._append_event(event_payload, dedupe_key=(event_payload["status"], packet_timestamp))
        self._write_debug(event_payload)
        return result


class HopeJrIsaacUpdateLoop:
    def __init__(
        self,
        controller: HopeJrSimIkController,
        *,
        apply_to_stage: bool,
        interval_s: float,
        reset_targets_on_stop: bool = True,
        reset_target_value_deg: float = 0.0,
    ):
        self.controller = controller
        self.apply_to_stage = apply_to_stage
        self.interval_s = interval_s
        self.reset_targets_on_stop = reset_targets_on_stop
        self.reset_target_value_deg = reset_target_value_deg
        self._subscription = None
        self._last_tick_time = 0.0
        self._last_status = None
        self._last_wait_seconds = None
        self._status_ui = HopeJrTeleopStatusUi()

    def _refresh_status_window(self) -> None:
        self._status_ui.update(self.controller, self.controller.last_debug_payload)

    def _on_update(self, _event: object) -> None:
        now = time.monotonic()
        if now - self._last_tick_time < self.interval_s:
            return
        self._last_tick_time = now
        try:
            result = self.controller.solve_once(apply_to_stage=self.apply_to_stage)
            debug_payload = self.controller.last_debug_payload or {}
            status = debug_payload.get("status")
            self._refresh_status_window()

            if status == "waiting_for_anchor":
                remaining = max(0.0, float(debug_payload.get("anchor_ready_time", 0.0)) - float(debug_payload.get("now", 0.0)))
                remaining_seconds = int(remaining + 0.999)
                if self._last_status != status or self._last_wait_seconds != remaining_seconds:
                    print(f"Hope Jr teleop: capture starts in {remaining_seconds}s")
                self._last_status = status
                self._last_wait_seconds = remaining_seconds
                return

            if result is not None:
                if result.get("status") == "reset_to_neutral":
                    print("Hope Jr teleop: reset to neutral")
                    self._last_status = "reset_to_neutral"
                    self._last_wait_seconds = None
                    return
                if self._last_status == "waiting_for_anchor":
                    print("Hope Jr teleop: anchor captured, teleop live")
                elif self._last_status != "applied":
                    print("Hope Jr teleop: receiving packets")
                self._last_status = "applied"
                self._last_wait_seconds = None
                return

            if status == "ignored":
                reason = debug_payload.get("reason")
                grip = debug_payload.get("grip")
                grip_threshold = debug_payload.get("grip_threshold")
                if reason == "packet_not_usable" and grip is not None and grip_threshold is not None and float(grip) < float(grip_threshold):
                    if self._last_status != "ignored:grip_threshold":
                        print(f"Hope Jr teleop: hold grip to move (current={grip:.2f}, required={grip_threshold:.2f})")
                    self._last_status = "ignored:grip_threshold"
                    self._last_wait_seconds = None
                    return
                if self._last_status != f"ignored:{reason}":
                    print(f"Hope Jr teleop: waiting for usable packet ({reason})")
                self._last_status = f"ignored:{reason}"
                self._last_wait_seconds = None
        except Exception as exc:
            print(f"Hope Jr IK update error: {exc}")

    def start(self) -> "HopeJrIsaacUpdateLoop":
        import omni.kit.app

        app = omni.kit.app.get_app()
        self._subscription = app.get_update_event_stream().create_subscription_to_pop(
            self._on_update,
            name="HopeJrSimIkController",
        )
        print(f"Hope Jr IK controller subscribed to Isaac update stream at {self.interval_s:.3f}s interval")
        return self

    def stop(self) -> None:
        self._subscription = None
        if self.reset_targets_on_stop:
            try:
                self.controller.reset_target_positions(self.reset_target_value_deg, reset_joint_state=True)
            except Exception as exc:
                print(f"Hope Jr IK controller target reset error: {exc}")
        print("Hope Jr IK controller unsubscribed from Isaac update stream")


def _parse_position_axes(value: str) -> tuple[int, int, int]:
    axis_map = {"x": 0, "y": 1, "z": 2}
    cleaned = value.strip().lower()
    if len(cleaned) != 3 or any(ch not in axis_map for ch in cleaned):
        raise ValueError(f"Invalid quest position axes mapping: {value}")
    return tuple(axis_map[ch] for ch in cleaned)


def build_controller_from_args(args: argparse.Namespace) -> HopeJrSimIkController:
    return HopeJrSimIkController(
        lerobot_repo=args.lerobot_repo,
        packet_path=args.packet_path,
        joint_root_path=args.joint_root_path,
        position_scale=args.position_scale,
        world_offset=np.asarray(args.world_offset, dtype=float),
        world_rotate_xyz_deg=np.asarray(args.world_rotate_xyz, dtype=float),
        quest_position_axes=_parse_position_axes(args.quest_position_axes),
        quest_position_signs=np.asarray(args.quest_position_signs, dtype=float),
        position_only=args.position_only,
        debug_path=args.debug_path,
        use_udp=args.use_udp,
        udp_listen_host=args.udp_listen_host,
        udp_listen_port=args.udp_listen_port,
        teleop_debug_root=args.teleop_debug_root,
        show_teleop_debug=args.show_teleop_debug,
        anchor_delay_s=args.anchor_delay_s,
        grip_threshold=args.grip_threshold,
        event_log_path=args.event_log_path,
        quest_deadband_m=args.quest_deadband_m,
        packet_stale_timeout_s=args.packet_stale_timeout_s,
        end_effector_path=args.end_effector_path,
        write_joint_state_directly=args.write_joint_state_directly,
        active_joint_names=tuple(args.active_joint_names) if getattr(args, 'active_joint_names', None) else None,
        inactive_joint_behavior=args.inactive_joint_behavior,
    )


def start_script_editor_loop(
    *,
    lerobot_repo: str | Path = DEFAULT_LEROBOT_REPO,
    packet_path: str | Path = DEFAULT_PACKET_PATH,
    joint_root_path: str = DEFAULT_JOINT_ROOT_PATH,
    position_scale: float = 1.0,
    world_offset: list[float] | tuple[float, float, float] = (0.0, 0.0, 0.0),
    world_rotate_xyz: list[float] | tuple[float, float, float] = (0.0, 0.0, 0.0),
    quest_position_axes: str = "xyz",
    quest_position_signs: list[float] | tuple[float, float, float] = (1.0, 1.0, 1.0),
    position_only: bool = True,
    debug_path: str | Path = DEFAULT_DEBUG_PATH,
    use_udp: bool = True,
    udp_listen_host: str = DEFAULT_UDP_LISTEN_HOST,
    udp_listen_port: int = DEFAULT_UDP_LISTEN_PORT,
    teleop_debug_root: str = DEFAULT_TELEOP_DEBUG_ROOT,
    show_teleop_debug: bool = True,
    anchor_delay_s: float = 3.0,
    grip_threshold: float = 0.25,
    event_log_path: str | Path = DEFAULT_EVENT_LOG_PATH,
    quest_deadband_m: float = 0.01,
    packet_stale_timeout_s: float = DEFAULT_PACKET_STALE_TIMEOUT_S,
    end_effector_path: str = DEFAULT_END_EFFECTOR_PATH,
    write_joint_state_directly: bool = False,
    active_joint_names: tuple[str, ...] | list[str] | None = None,
    inactive_joint_behavior: str = "neutral",
    interval_s: float = 0.05,
    dry_run: bool = False,
    consume_only_new: bool = True,
    reset_targets_on_stop: bool = True,
    reset_target_value_deg: float = 0.0,
) -> HopeJrIsaacUpdateLoop:
    global _ACTIVE_LOOP
    stop_script_editor_loop()
    controller = HopeJrSimIkController(
        lerobot_repo=Path(lerobot_repo),
        packet_path=Path(packet_path),
        joint_root_path=joint_root_path,
        position_scale=position_scale,
        world_offset=np.asarray(world_offset, dtype=float),
        world_rotate_xyz_deg=np.asarray(world_rotate_xyz, dtype=float),
        quest_position_axes=_parse_position_axes(quest_position_axes),
        quest_position_signs=np.asarray(quest_position_signs, dtype=float),
        position_only=position_only,
        debug_path=Path(debug_path),
        use_udp=use_udp,
        udp_listen_host=udp_listen_host,
        udp_listen_port=udp_listen_port,
        teleop_debug_root=teleop_debug_root,
        show_teleop_debug=show_teleop_debug,
        anchor_delay_s=anchor_delay_s,
        grip_threshold=grip_threshold,
        event_log_path=Path(event_log_path),
        quest_deadband_m=quest_deadband_m,
        packet_stale_timeout_s=packet_stale_timeout_s,
        end_effector_path=end_effector_path,
        write_joint_state_directly=write_joint_state_directly,
        active_joint_names=tuple(active_joint_names) if active_joint_names else None,
        inactive_joint_behavior=inactive_joint_behavior,
    )
    try:
        controller.event_log_path.unlink(missing_ok=True)
    except Exception:
        pass
    if consume_only_new:
        controller.minimum_packet_timestamp = time.time()
    _ACTIVE_LOOP = HopeJrIsaacUpdateLoop(
        controller,
        apply_to_stage=not dry_run,
        interval_s=interval_s,
        reset_targets_on_stop=reset_targets_on_stop,
        reset_target_value_deg=reset_target_value_deg,
    ).start()
    return _ACTIVE_LOOP


def stop_script_editor_loop() -> None:
    global _ACTIVE_LOOP
    if _ACTIVE_LOOP is not None:
        _ACTIVE_LOOP.stop()
        _ACTIVE_LOOP = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="First-pass Hope Jr Quest -> Sim IK controller")
    parser.add_argument("--lerobot-repo", type=Path, default=DEFAULT_LEROBOT_REPO)
    parser.add_argument("--packet-path", type=Path, default=DEFAULT_PACKET_PATH)
    parser.add_argument("--joint-root-path", default=DEFAULT_JOINT_ROOT_PATH)
    parser.add_argument("--position-scale", type=float, default=1.0)
    parser.add_argument("--world-offset", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--world-rotate-xyz", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--quest-position-axes", default="xyz")
    parser.add_argument("--quest-position-signs", nargs=3, type=float, default=[1.0, 1.0, 1.0])
    parser.add_argument("--position-only", action="store_true")
    parser.add_argument("--debug-path", type=Path, default=DEFAULT_DEBUG_PATH)
    parser.add_argument("--use-udp", action="store_true")
    parser.add_argument("--udp-listen-host", default=DEFAULT_UDP_LISTEN_HOST)
    parser.add_argument("--udp-listen-port", type=int, default=DEFAULT_UDP_LISTEN_PORT)
    parser.add_argument("--teleop-debug-root", default=DEFAULT_TELEOP_DEBUG_ROOT)
    parser.add_argument("--show-teleop-debug", action="store_true")
    parser.add_argument("--anchor-delay-s", type=float, default=3.0)
    parser.add_argument("--grip-threshold", type=float, default=0.25)
    parser.add_argument("--event-log-path", type=Path, default=DEFAULT_EVENT_LOG_PATH)
    parser.add_argument("--quest-deadband-m", type=float, default=0.01)
    parser.add_argument("--packet-stale-timeout-s", type=float, default=DEFAULT_PACKET_STALE_TIMEOUT_S)
    parser.add_argument("--end-effector-path", default=DEFAULT_END_EFFECTOR_PATH)
    parser.add_argument("--write-joint-state-directly", action="store_true")
    parser.add_argument("--active-joint-names", nargs="*", default=None)
    parser.add_argument("--inactive-joint-behavior", choices=("neutral", "hold"), default="neutral")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--isaac-update-loop", action="store_true")
    parser.add_argument("--interval", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    controller = build_controller_from_args(args)

    if args.isaac_update_loop:
        HopeJrIsaacUpdateLoop(controller, apply_to_stage=not args.dry_run, interval_s=args.interval).start()
        return

    if args.watch:
        print(f"watching {args.packet_path}")
        print(f"joint root path: {args.joint_root_path}")
        while True:
            result = controller.solve_once(apply_to_stage=not args.dry_run)
            if result is not None:
                print(json.dumps(result, indent=2))
            time.sleep(args.interval)

    result = controller.solve_once(apply_to_stage=not args.dry_run)
    if result is None:
        print("no normalized Quest packet available yet")
        return
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
