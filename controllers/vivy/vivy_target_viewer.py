from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


SIM_CONFIG_PATH = Path("/home/viaan/huggingface/lerobot/src/lerobot/robots/vivy/vivy_global_config.json")
KINEMATICS_PATH = Path("/home/viaan/huggingface/lerobot/src/lerobot/robots/vivy/vivy_arm_kinematics.py")
QUEST_TELEOP_MAPPER_PATH = Path("/home/viaan/vivy_hopejr_sim/controllers/quest_teleop_mapper.py")
TELEOP_STATE_PATH = Path("/home/viaan/huggingface/lerobot/src/lerobot/robots/vivy/fanout/teleop_state.py")
TELEOP_DEBUG_VISUALS_PATH = Path("/home/viaan/vivy_hopejr_sim/ui/teleop_debug_visuals.py")
STAGE_IO_PATH = Path("/home/viaan/vivy_hopejr_sim/controllers/stage_io.py")
VIVY_SIDE_PANEL_PATH = Path("/home/viaan/vivy_hopejr_sim/ui/vivy/vivy_side_panel.py")
VIVY_FLOW_PANEL_PATH = Path("/home/viaan/vivy_hopejr_sim/ui/vivy/vivy_flow_panel.py")
VIVY_FLOW_DETAIL_PANEL_PATH = Path("/home/viaan/vivy_hopejr_sim/ui/vivy/vivy_flow_detail_panel.py")
FLOW_CONTROL_PATH = Path("/tmp/vivy_flow_control.json")
STAGE_FEEDBACK_PATH = Path("/tmp/vivy_stage_feedback.json")
REAL_FEEDBACK_PATH = Path("/tmp/vivy_real_feedback.json")
SIM_WRITE_DEBUG_PATH = Path("/tmp/vivy_sim_write_debug.json")
SIM_WRITE_EVENTS_PATH = Path("/tmp/vivy_sim_write_events.ndjson")

_ACTIVE_LOOP = None


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_sim_config() -> dict:
    return json.loads(SIM_CONFIG_PATH.read_text(encoding="utf-8"))


def _load_kinematics_class():
    module = _load_module("vivy_target_viewer_kinematics", KINEMATICS_PATH)
    return getattr(module, "VivyArmKinematics", module.HopeJrArmKinematics)


def _load_kinematics_make_pose():
    module = _load_module("vivy_target_viewer_kinematics_make_pose", KINEMATICS_PATH)
    return module.make_pose


def _load_quest_teleop_mapper_class():
    module = _load_module("vivy_target_viewer_quest_teleop_mapper", QUEST_TELEOP_MAPPER_PATH)
    return module.QuestTeleopMapper


def _load_shared_signal_helpers():
    module = _load_module("vivy_target_viewer_teleop_state", TELEOP_STATE_PATH)
    return module.DEFAULT_TELEOP_STATE_PATH, module.read_teleop_state


def _load_visuals_class():
    module = _load_module("vivy_target_viewer_visuals", TELEOP_DEBUG_VISUALS_PATH)
    return module.TeleopDebugVisuals


def _load_stage_io_class():
    module = _load_module("vivy_target_viewer_stage_io", STAGE_IO_PATH)
    return module.HopeJrStageIo


def _load_side_panel_class():
    module = _load_module("vivy_side_panel", VIVY_SIDE_PANEL_PATH)
    return module.VivySidePanel


def _load_flow_panel_class():
    module = _load_module("vivy_flow_panel", VIVY_FLOW_PANEL_PATH)
    return module.VivyFlowPanel


def _load_flow_detail_panel_class():
    module = _load_module("vivy_flow_detail_panel", VIVY_FLOW_DETAIL_PANEL_PATH)
    return module.VivyFlowDetailPanel


def _read_flow_control(path: Path = FLOW_CONTROL_PATH) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_stage_feedback(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _write_sim_write_debug(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _append_sim_write_event(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _load_json_file(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _parse_position_axes(value: str) -> tuple[int, int, int]:
    axis_map = {"x": 0, "y": 1, "z": 2}
    cleaned = str(value).strip().lower()
    if len(cleaned) != 3 or any(ch not in axis_map for ch in cleaned):
        return (0, 1, 2)
    return tuple(axis_map[ch] for ch in cleaned)


class VivyTargetViewer:
    def __init__(self, *, signal_path: str | Path | None = None, interval_s: float = 0.05):
        default_signal_path, read_teleop_state = _load_shared_signal_helpers()
        VivyArmKinematics = _load_kinematics_class()
        make_pose = _load_kinematics_make_pose()
        QuestTeleopMapper = _load_quest_teleop_mapper_class()
        HopeJrStageIo = _load_stage_io_class()
        TeleopDebugVisuals = _load_visuals_class()
        VivySidePanel = _load_side_panel_class()
        VivyFlowPanel = _load_flow_panel_class()
        VivyFlowDetailPanel = _load_flow_detail_panel_class()
        self._read_teleop_state = read_teleop_state
        self.signal_path = Path(default_signal_path if signal_path is None else signal_path)
        self.interval_s = float(interval_s)
        self._last_tick_time = 0.0
        self._subscription = None
        self._last_signal_timestamp = None
        self._last_applied_feedback_timestamp: float | None = None
        self._last_applied_command_timestamp: float | None = None

        self.sim_config = _load_sim_config()
        controller_defaults = dict(self.sim_config.get("controller_defaults") or {})
        self.end_effector_path = controller_defaults.get("end_effector_path", "/World/JointTest/RightForearm/EndEffector")
        self.teleop_debug_root = controller_defaults.get("teleop_debug_root", "/World/JointTest/TeleopDebug")
        self.articulation_root_path = controller_defaults.get("articulation_root_path", "/World/JointTest")
        self.joint_root_path = controller_defaults.get("joint_root_path", "/World/JointTest/Joints")
        self.world_offset = np.asarray(controller_defaults.get("world_offset", [0.0, 0.0, 0.0]), dtype=float)
        self.real_feedback_max_age_s = float(controller_defaults.get("packet_stale_timeout_s", 0.75))
        default_chain = str(self.sim_config.get("default_ik_chain", "right_arm"))
        chain_config = (self.sim_config.get("ik_chains") or {}).get(default_chain) or {}
        ik_joint_names = list(chain_config["joint_names"])
        controlled_joint_names = list(chain_config["controlled_joint_names"])
        self.ik_joint_names = list(ik_joint_names)
        self.joint_names = list(controlled_joint_names)
        self.stage_joint_names = list(self.sim_config["joints"].keys())
        neutral_deg = np.asarray([float(self.sim_config["joints"][name]["neutral_deg"]) for name in ik_joint_names], dtype=float)
        self.neutral_joint_positions_deg = np.asarray(
            [float(self.sim_config["joints"][name]["neutral_deg"]) for name in controlled_joint_names],
            dtype=float,
        )
        self.stage_neutral_joint_positions_deg = np.asarray(
            [float(self.sim_config["joints"][name]["neutral_deg"]) for name in self.stage_joint_names],
            dtype=float,
        )
        self.stage_min_joint_positions_deg = np.asarray(
            [float(self.sim_config["joints"][name]["min_deg"]) for name in self.stage_joint_names],
            dtype=float,
        )
        self.stage_max_joint_positions_deg = np.asarray(
            [float(self.sim_config["joints"][name]["max_deg"]) for name in self.stage_joint_names],
            dtype=float,
        )
        ik_chains = dict(self.sim_config.get("ik_chains") or {})
        left_chain_config = dict(ik_chains.get("left_arm") or {})
        left_end_effector_config = dict(left_chain_config.get("end_effector") or {})
        left_end_effector_path = str(left_end_effector_config.get("frame_path", "/World/JointTest/LeftForearm/EndEffector"))
        left_joint_names = list(left_chain_config.get("joint_names") or [])
        if not left_joint_names:
            left_joint_names = list(self.ik_joint_names)
        self.left_kinematics = VivyArmKinematics.from_json(SIM_CONFIG_PATH, chain_name="left_arm")
        self.left_position_scale = float(controller_defaults.get("position_scale", 1.0))
        self.left_world_offset = np.asarray(controller_defaults.get("world_offset", [0.0, 0.0, 0.0]), dtype=float)
        self.left_world_rotate_xyz = np.asarray(controller_defaults.get("world_rotate_xyz", [0.0, 0.0, 0.0]), dtype=float)
        self.left_quest_position_axes = _parse_position_axes(str(controller_defaults.get("quest_position_axes", "xyz")))
        self.left_quest_position_signs = np.asarray(controller_defaults.get("quest_position_signs", [1.0, 1.0, 1.0]), dtype=float)
        self.left_teleop_mapper = QuestTeleopMapper(
            position_scale=self.left_position_scale,
            world_offset=self.left_world_offset,
            world_rotate_xyz_deg=self.left_world_rotate_xyz,
            quest_position_axes=self.left_quest_position_axes,
            quest_position_signs=self.left_quest_position_signs,
            position_only=bool(controller_defaults.get("position_only", True)),
            anchor_delay_s=float(controller_defaults.get("anchor_delay_s", 1.0)),
            quest_deadband_m=float(controller_defaults.get("quest_deadband_m", 0.01)),
            make_pose=make_pose,
            hand_key="left_hand",
        )
        self.controlled_min_joint_positions_deg = np.asarray(
            [float(self.sim_config["joints"][name]["min_deg"]) for name in self.joint_names],
            dtype=float,
        )
        self.controlled_max_joint_positions_deg = np.asarray(
            [float(self.sim_config["joints"][name]["max_deg"]) for name in self.joint_names],
            dtype=float,
        )
        self._controlled_to_stage_indices = np.asarray(
            [self.stage_joint_names.index(name) for name in self.joint_names],
            dtype=np.int64,
        )
        self.kinematics = VivyArmKinematics.from_json(SIM_CONFIG_PATH, chain_name=default_chain)
        self.model_neutral_pose = self.kinematics.forward_kinematics(neutral_deg)
        self.model_to_stage_transform = np.eye(4)
        self._anchor_model_to_stage_transform = np.eye(4)
        self.left_stage_io = HopeJrStageIo(
            articulation_root_path=self.articulation_root_path,
            joint_root_path=self.joint_root_path,
            end_effector_path=left_end_effector_path,
            joint_names=left_joint_names,
        )
        self._stage_anchor_pose = None
        self._last_waiting_for_anchor = True
        self.stage_io = HopeJrStageIo(
            articulation_root_path=self.articulation_root_path,
            joint_root_path=self.joint_root_path,
            end_effector_path=self.end_effector_path,
            joint_names=self.joint_names,
        )
        self.visuals = TeleopDebugVisuals(teleop_debug_root=self.teleop_debug_root, enabled=True)
        self.side_panel = VivySidePanel()
        self.flow_panel = VivyFlowPanel()
        self.flow_detail_panel = VivyFlowDetailPanel()
        self._ensure_panels_created()
        self._joint_write_ready = False
        self._last_joint_write_warn_time = 0.0
        self._waiting_anchor_neutral_applied = False
        self._last_panel_signal_timestamp: float | None = None
        self._last_panel_signal_arrival_time: float | None = None
        self._last_panel_real_feedback_timestamp: float | None = None
        self._last_panel_real_feedback_arrival_time: float | None = None
        self._last_bus_hz: float | None = None
        self._last_stage_joint_positions_deg: np.ndarray | None = None

    def _ensure_panels_created(self) -> None:
        for panel in (self.side_panel, self.flow_panel, self.flow_detail_panel):
            try:
                panel._ensure_window()
            except Exception:
                pass


    def _read_real_feedback(self) -> dict[str, object] | None:
        payload = _load_json_file(REAL_FEEDBACK_PATH)
        if not isinstance(payload, dict):
            return None
        timestamp = payload.get("timestamp")
        try:
            feedback_age_s = time.time() - float(timestamp)
        except (TypeError, ValueError):
            return None
        if feedback_age_s < 0.0 or feedback_age_s > self.real_feedback_max_age_s:
            return None
        return payload

    def _read_real_feedback_joint_positions_deg(self, payload: dict[str, object] | None) -> np.ndarray | None:
        if not isinstance(payload, dict):
            return None
        joint_positions_deg: list[float] = []
        for joint_name in self.joint_names:
            value = payload.get(f"{joint_name}.pos_deg")
            if value is None:
                value = payload.get(f"{joint_name}.pos")
            if value is None:
                return None
            try:
                joint_positions_deg.append(float(value))
            except (TypeError, ValueError):
                return None
        return np.asarray(joint_positions_deg, dtype=float)

    def _inject_real_feedback_rows(
        self,
        payload: dict[str, object],
        real_feedback: dict[str, object] | None,
    ) -> dict[str, object]:
        merged_payload = dict(payload)
        merged_payload["teleop_hz"] = self._compute_teleop_hz(payload)
        merged_payload["bus_hz"] = self._compute_bus_hz(real_feedback)
        rows = list(payload.get("joint_display_rows") or [])
        merged_payload["real_feedback_live"] = bool(real_feedback is not None)
        merged_payload["real_feedback_status"] = "live" if real_feedback is not None else "stale"
        if not rows:
            return merged_payload

        merged_rows: list[dict[str, object]] = []
        for index, row in enumerate(rows):
            merged_row = dict(row)
            if real_feedback is not None and index < len(self.joint_names):
                joint_name = self.joint_names[index]
                raw_value = real_feedback.get(f"{joint_name}.servo_raw")
                deg_value = real_feedback.get(f"{joint_name}.pos_deg")
                merged_row["actual_raw"] = "-" if raw_value is None else str(int(raw_value))
                merged_row["actual_deg"] = "-" if deg_value is None else f"{float(deg_value):7.2f}"
                try:
                    target_deg = float(row.get("target_deg"))
                    actual_deg = float(deg_value)
                    merged_row["error_deg"] = f"{(target_deg - actual_deg):+7.2f}"
                except (TypeError, ValueError):
                    merged_row["error_deg"] = "-"
            else:
                merged_row["actual_raw"] = "-"
                merged_row["actual_deg"] = "-"
                merged_row["error_deg"] = "-"
            merged_rows.append(merged_row)
        merged_payload["joint_display_rows"] = merged_rows
        return merged_payload

    def _compute_teleop_hz(self, payload: dict[str, object]) -> float | None:
        timestamp = payload.get("timestamp")
        try:
            timestamp_f = float(timestamp)
        except (TypeError, ValueError):
            return None
        arrival_now = time.monotonic()
        hz = None
        if self._last_panel_signal_timestamp is not None and self._last_panel_signal_arrival_time is not None:
            dt = timestamp_f - self._last_panel_signal_timestamp
            if dt > 1e-6:
                hz = 1.0 / dt
        self._last_panel_signal_timestamp = timestamp_f
        self._last_panel_signal_arrival_time = arrival_now
        return hz

    def _compute_bus_hz(self, real_feedback: dict[str, object] | None) -> float | None:
        if not isinstance(real_feedback, dict):
            return None
        timestamp = real_feedback.get("timestamp")
        try:
            timestamp_f = float(timestamp)
        except (TypeError, ValueError):
            return None
        arrival_now = time.monotonic()
        if timestamp_f == self._last_panel_real_feedback_timestamp:
            if self._last_panel_real_feedback_arrival_time is None:
                return self._last_bus_hz
            age_s = arrival_now - self._last_panel_real_feedback_arrival_time
            return self._last_bus_hz if age_s <= self.real_feedback_max_age_s else None

        hz = self._last_bus_hz
        if self._last_panel_real_feedback_timestamp is not None:
            dt = timestamp_f - self._last_panel_real_feedback_timestamp
            if dt > 1e-6:
                hz = 1.0 / dt

        self._last_panel_real_feedback_timestamp = timestamp_f
        self._last_panel_real_feedback_arrival_time = arrival_now
        self._last_bus_hz = hz
        return hz

    @staticmethod
    def _coerce_timestamp(payload: dict[str, object] | None) -> float | None:
        if not isinstance(payload, dict):
            return None
        value = payload.get("timestamp")
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _write_sim_joint_positions(self, stage, joint_positions_deg: np.ndarray, *, update_state: bool) -> None:
        values = np.asarray(joint_positions_deg, dtype=float)
        input_values = values.copy()
        if values.shape == (len(self.stage_joint_names),):
            values = values[self._controlled_to_stage_indices]
        values = np.clip(np.asarray(values, dtype=float), self.controlled_min_joint_positions_deg, self.controlled_max_joint_positions_deg)
        debug_payload = {
            "timestamp": time.time(),
            "input_joint_count": int(input_values.size),
            "commanded_joint_count": len(self.joint_names),
            "commanded_joint_names": list(self.joint_names),
            "input_joint_positions_deg": [float(value) for value in input_values.reshape(-1)],
            "commanded_joint_positions_deg": [float(value) for value in np.asarray(values, dtype=float).reshape(-1)],
            "update_state": bool(update_state),
        }
        try:
            self.stage_io.write_joint_targets_deg(stage, values)
            if update_state:
                self.stage_io.write_joint_state_deg(stage, values)
            self._joint_write_ready = True
            debug_payload["success"] = True
        except Exception as exc:
            exc_text = str(exc)
            debug_payload["success"] = False
            debug_payload["error"] = exc_text
            now = time.monotonic()
            if "Articulation unavailable" in exc_text:
                self._joint_write_ready = False
            elif now - self._last_joint_write_warn_time > 2.0:
                print(f"Vivy target viewer joint write failed: {exc}")
                self._last_joint_write_warn_time = now
        debug_payload["stage_io_debug"] = getattr(self.stage_io, "last_stage_dls_debug", None)
        try:
            _write_sim_write_debug(SIM_WRITE_DEBUG_PATH, debug_payload)
            _append_sim_write_event(SIM_WRITE_EVENTS_PATH, debug_payload)
        except Exception:
            pass

    def _read_stage_joint_target_attrs_deg(self, stage) -> np.ndarray | None:
        if stage is None:
            return None
        targets: list[float] = []
        for joint_name in self.stage_joint_names:
            prim = stage.GetPrimAtPath(f"{self.joint_root_path}/{joint_name}")
            if not prim.IsValid():
                return None
            attr = prim.GetAttribute("drive:angular:physics:targetPosition")
            value = attr.Get() if attr.IsValid() else None
            if value is None:
                state_attr = prim.GetAttribute("state:angular:physics:position")
                value = state_attr.Get() if state_attr.IsValid() else None
            if value is None:
                return None
            try:
                targets.append(float(value))
            except (TypeError, ValueError):
                return None
        arr = np.asarray(targets, dtype=float)
        if arr.shape != (len(self.stage_joint_names),) or not np.all(np.isfinite(arr)):
            return None
        return np.clip(arr, self.stage_min_joint_positions_deg, self.stage_max_joint_positions_deg)

    def _read_stage(self):
        import omni.usd

        return omni.usd.get_context().get_stage()

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
            return np.array(world_transform, dtype=float).T
        except Exception:
            return None

    def _maybe_refresh_transform(self, stage) -> np.ndarray | None:
        stage_pose = self._read_stage_end_effector_pose(stage)
        if stage_pose is None:
            return None
        try:
            self.model_to_stage_transform = stage_pose @ np.linalg.inv(self.model_neutral_pose)
        except np.linalg.LinAlgError:
            self.model_to_stage_transform = np.eye(4)
        return stage_pose

    @staticmethod
    def _transform_from_rt(position: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
        transform = np.eye(4)
        transform[:3, :3] = np.asarray(rotation_matrix, dtype=float)
        transform[:3, 3] = np.asarray(position, dtype=float)
        return transform

    @staticmethod
    def _quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
        quat_wxyz = np.asarray(quat_wxyz, dtype=float)
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=float)
        return Rotation.from_quat(quat_xyzw).as_matrix()

    @staticmethod
    def _axis_vector(axis: str) -> np.ndarray:
        sign = -1.0 if str(axis).startswith('-') else 1.0
        basis = str(axis).lstrip('-').upper()
        vectors = {
            'X': np.array([1.0, 0.0, 0.0], dtype=float),
            'Y': np.array([0.0, 1.0, 0.0], dtype=float),
            'Z': np.array([0.0, 0.0, 1.0], dtype=float),
        }
        return sign * vectors.get(basis, vectors['Z'])

    def _build_pitch_visual(self, stage) -> dict[str, np.ndarray] | None:
        if stage is None:
            return None
        try:
            from pxr import UsdGeom
            pitch_index = self.joint_names.index('right_shoulder_pitch')
            joint = self.kinematics.joints[pitch_index]
            xform_cache = UsdGeom.XformCache()
            parent_prim = stage.GetPrimAtPath(joint.parent_body)
            child_prim = stage.GetPrimAtPath(joint.child_body)
            if not parent_prim.IsValid() or not child_prim.IsValid():
                return None
            parent_body_transform = np.array(xform_cache.GetLocalToWorldTransform(parent_prim), dtype=float).T
            child_body_transform = np.array(xform_cache.GetLocalToWorldTransform(child_prim), dtype=float).T
            parent_frame = parent_body_transform @ self._transform_from_rt(
                joint.local_pos0,
                self._quat_wxyz_to_matrix(joint.local_rot0_quat_wxyz),
            )
            child_frame = child_body_transform @ self._transform_from_rt(
                joint.local_pos1,
                self._quat_wxyz_to_matrix(joint.local_rot1_quat_wxyz),
            )
            preview_offset_dir = parent_frame[:3, 1] + parent_frame[:3, 2]
            preview_offset_norm = float(np.linalg.norm(preview_offset_dir))
            if preview_offset_norm <= 1e-9:
                preview_offset_dir = np.array([0.0, 0.0, 1.0], dtype=float)
            else:
                preview_offset_dir = preview_offset_dir / preview_offset_norm
            child_frame_offset = child_frame.copy()
            child_frame_offset[:3, 3] = child_frame_offset[:3, 3] + preview_offset_dir * 0.03
            axis_world = parent_frame[:3, :3] @ self._axis_vector(joint.axis)
            return {
                'parent_frame': parent_frame,
                'child_frame': child_frame_offset,
                'child_frame_raw': child_frame,
                'axis_world': axis_world,
                'preview_offset_dir': preview_offset_dir,
            }
        except Exception:
            return None

    @staticmethod
    def _extract_left_hand_state(payload: dict) -> dict | None:
        normalized = payload.get("normalized")
        if isinstance(normalized, dict):
            left_hand = normalized.get("left_hand")
            if isinstance(left_hand, dict):
                return left_hand
        parsed_message = payload.get("parsed_message")
        if isinstance(parsed_message, dict):
            for key in ("left_hand", "left", "leftController", "LeftHand"):
                left_hand = parsed_message.get(key)
                if isinstance(left_hand, dict):
                    return left_hand
        return None

    def _on_update(self, _event: object) -> None:
        now = time.monotonic()
        if now - self._last_tick_time < self.interval_s:
            return
        self._last_tick_time = now

        stage = self._read_stage()
        stage_pose = self._maybe_refresh_transform(stage)
        if stage is None or stage_pose is None:
            return

        payload = self._read_teleop_state(self.signal_path)
        if not isinstance(payload, dict):
            return
        flow_control = _read_flow_control()
        real_feedback = self._read_real_feedback()
        real_joint_positions_deg = self._read_real_feedback_joint_positions_deg(real_feedback)
        panel_payload = self._inject_real_feedback_rows(payload, real_feedback)
        try:
            self.flow_panel.update(payload, flow_control)
        except Exception:
            pass
        try:
            self.flow_detail_panel.update(payload, flow_control)
        except Exception:
            pass

        stage_joint_positions_deg = self.stage_io.read_stage_joint_positions_deg(stage)
        if stage_joint_positions_deg is not None:
            self._last_stage_joint_positions_deg = np.asarray(stage_joint_positions_deg, dtype=float).copy()
            controlled_stage_joint_positions_deg = np.asarray(stage_joint_positions_deg, dtype=float)
            try:
                _write_stage_feedback(
                    STAGE_FEEDBACK_PATH,
                    {
                        "timestamp": time.time(),
                        "joint_names": list(self.joint_names),
                        **{
                            f"{joint_name}.pos_deg": float(controlled_stage_joint_positions_deg[index])
                            for index, joint_name in enumerate(self.joint_names)
                        },
                    },
                )
            except Exception:
                pass

        waiting_for_anchor = bool(payload.get("waiting_for_anchor", True))
        if waiting_for_anchor:
            self._stage_anchor_pose = None
            self._anchor_model_to_stage_transform = np.eye(4)
        elif self._stage_anchor_pose is None or self._last_waiting_for_anchor:
            self._stage_anchor_pose = stage_pose.copy()
            target_pose_model = payload.get("target_pose_model")
            if target_pose_model is not None:
                try:
                    target_pose_model_arr = np.asarray(target_pose_model, dtype=float)
                    if target_pose_model_arr.shape == (4, 4):
                        self._anchor_model_to_stage_transform = self._stage_anchor_pose @ np.linalg.inv(
                            target_pose_model_arr
                        )
                except Exception:
                    self._anchor_model_to_stage_transform = np.eye(4)
        self._last_waiting_for_anchor = waiting_for_anchor

        signal_timestamp = self._coerce_timestamp(payload)
        feedback_timestamp = self._coerce_timestamp(real_feedback)
        teleop_changed = signal_timestamp is not None and signal_timestamp != self._last_signal_timestamp
        feedback_changed = feedback_timestamp is not None and feedback_timestamp != self._last_applied_feedback_timestamp

        follow_target_enabled = bool(payload.get("follow_target_enabled", False))
        stage_write_blocked = waiting_for_anchor or not follow_target_enabled
        self._waiting_anchor_neutral_applied = False
        joint_targets_deg = payload.get("current_joint_targets_deg")
        if not stage_write_blocked:
            if real_joint_positions_deg is not None:
                if feedback_changed:
                    self._write_sim_joint_positions(stage, np.asarray(real_joint_positions_deg, dtype=float), update_state=False)
                    self._last_applied_feedback_timestamp = feedback_timestamp
            elif (
                teleop_changed
                and isinstance(joint_targets_deg, list)
                and len(joint_targets_deg) == len(self.joint_names)
            ):
                self._write_sim_joint_positions(stage, np.asarray(joint_targets_deg, dtype=float), update_state=False)
                self._last_applied_command_timestamp = signal_timestamp

        if teleop_changed:
            self._last_signal_timestamp = signal_timestamp
        else:
            return

        stage_anchor_position = stage_pose[:3, 3] if self._stage_anchor_pose is None else self._stage_anchor_pose[:3, 3]
        conditioned_delta_world = np.asarray(
            payload.get("conditioned_position_delta_world") or [0.0, 0.0, 0.0],
            dtype=float,
        )
        quest_mapped_position = np.asarray(
            stage_anchor_position + conditioned_delta_world,
            dtype=float,
        )
        sim_target_position = np.asarray(
            quest_mapped_position + self.world_offset,
            dtype=float,
        )
        target_pose_model = payload.get("target_pose_model")
        if target_pose_model is not None:
            try:
                target_pose_model_arr = np.asarray(target_pose_model, dtype=float)
                if target_pose_model_arr.shape == (4, 4):
                    sim_target_position = (
                        self._anchor_model_to_stage_transform @ target_pose_model_arr
                    )[:3, 3] + self.world_offset
            except Exception:
                pass
        left_map_result = None
        left_hand_payload = None
        if stage is not None:
            left_hand = self._extract_left_hand_state(payload)
            if isinstance(left_hand, dict):
                left_hand_payload = left_hand
                left_joint_targets_deg = self.left_stage_io.read_current_joint_targets_deg(stage)
                if left_joint_targets_deg is not None:
                    left_current_sim_pose = self.left_kinematics.forward_kinematics(
                        np.asarray(left_joint_targets_deg, dtype=float)
                    )
                    left_stage_pose = self.left_stage_io.read_stage_end_effector_pose(stage)
                    left_map_result = self.left_teleop_mapper.map_packet(
                        {"normalized": {"left_hand": left_hand}},
                        current_sim_pose=left_current_sim_pose,
                        current_stage_pose=left_stage_pose,
                        anchor_joint_targets_deg=np.asarray(left_joint_targets_deg, dtype=float),
                    )
        payload["left_hand_state"] = left_hand_payload
        payload["left_follow_target_enabled"] = None if left_map_result is None else left_map_result.follow_target_enabled
        payload["left_waiting_for_anchor"] = None if left_map_result is None else left_map_result.waiting_for_anchor
        payload["left_anchor_position"] = None if left_map_result is None else left_map_result.quest_anchor_position
        panel_payload["left_hand_state"] = left_hand_payload
        panel_payload["left_follow_target_enabled"] = None if left_map_result is None else left_map_result.follow_target_enabled
        panel_payload["left_waiting_for_anchor"] = None if left_map_result is None else left_map_result.waiting_for_anchor
        panel_payload["left_anchor_position"] = None if left_map_result is None else left_map_result.quest_anchor_position
        try:
            self.side_panel.update(panel_payload)
        except Exception:
            pass
        show_pitch_frames = bool(flow_control.get("show_pitch_frames", False))
        pitch_visual = self._build_pitch_visual(stage) if show_pitch_frames else None
        self.visuals.enabled = bool(flow_control.get("sim_view_enabled", True))
        self.visuals.update(
            stage,
            quest_anchor_position=np.asarray(payload.get("quest_anchor_position") or [0.0, 0.0, 0.0], dtype=float),
            quest_current_position=np.asarray((payload.get("hand_state") or {}).get("position") or [0.0, 0.0, 0.0], dtype=float),
            quest_mapped_position=quest_mapped_position,
            sim_target_position=sim_target_position,
            left_quest_mapped_position=None if left_map_result is None else left_map_result.quest_mapped_position_stage,
            left_sim_target_position=None if left_map_result is None else left_map_result.sim_target_position_stage,
            left_waiting_for_anchor=None if left_map_result is None else left_map_result.waiting_for_anchor,
            reference_position=stage_pose[:3, 3],
            actual_end_effector_position=stage_pose[:3, 3],
            actual_end_effector_pose=stage_pose,
            waiting_for_anchor=waiting_for_anchor,
            show_pitch_frames=show_pitch_frames,
            pitch_visual=pitch_visual,
        )

    def start(self):
        import omni.kit.app

        app = omni.kit.app.get_app()
        self._subscription = app.get_update_event_stream().create_subscription_to_pop(
            self._on_update,
            name="VivyTargetViewer",
        )
        print(f"Vivy target viewer subscribed at {self.interval_s:.3f}s interval")
        return self

    def stop(self) -> None:
        self._subscription = None
        print("Vivy target viewer unsubscribed")


def start_script_editor_loop(*, signal_path: str | Path | None = None, interval_s: float = 0.05):
    global _ACTIVE_LOOP
    stop_script_editor_loop()
    _ACTIVE_LOOP = VivyTargetViewer(signal_path=signal_path, interval_s=interval_s).start()
    return _ACTIVE_LOOP


def stop_script_editor_loop() -> None:
    global _ACTIVE_LOOP
    if _ACTIVE_LOOP is not None:
        _ACTIVE_LOOP.stop()
        _ACTIVE_LOOP = None
