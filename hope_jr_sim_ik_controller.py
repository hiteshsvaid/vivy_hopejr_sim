#!/usr/bin/env python3

import argparse
import importlib.util
import json
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


DEFAULT_LEROBOT_REPO = Path("/home/viaan/huggingface/lerobot")
DEFAULT_PACKET_PATH = Path("/tmp/hope_jr_quest_latest.json")
DEFAULT_IK_SPEC_PATH = DEFAULT_LEROBOT_REPO / "src/lerobot/robots/hope_jr/hope_jr_arm_ik_spec.json"
DEFAULT_KINEMATICS_MODULE_PATH = DEFAULT_LEROBOT_REPO / "src/lerobot/robots/hope_jr/hope_jr_arm_kinematics.py"
DEFAULT_JOINT_ROOT_PATH = "/World/JointTest/Joints"
DEFAULT_UDP_LISTEN_HOST = "127.0.0.1"
DEFAULT_UDP_LISTEN_PORT = 8766
DEFAULT_DEBUG_PATH = Path("/tmp/hope_jr_sim_ik_debug.json")
DEFAULT_TELEOP_DEBUG_ROOT = "/World/JointTest/TeleopDebug"
DEFAULT_EVENT_LOG_PATH = Path("/tmp/hope_jr_sim_ik_events.ndjson")
DEFAULT_STOP_TARGETS_DEG = {
    "right_elbow": 40.5,
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
    ):
        self.lerobot_repo = lerobot_repo
        self.packet_path = packet_path
        self.joint_root_path = joint_root_path.rstrip("/")
        self.position_scale = position_scale
        self.world_offset = world_offset
        self.world_rotation = Rotation.from_euler("XYZ", world_rotate_xyz_deg, degrees=True).as_matrix()
        self.quest_position_axes = quest_position_axes
        self.quest_position_signs = quest_position_signs
        self.position_only = position_only
        self.debug_path = debug_path
        self.use_udp = use_udp
        self.udp_listen_host = udp_listen_host
        self.udp_listen_port = udp_listen_port
        self.teleop_debug_root = teleop_debug_root.rstrip("/")
        self.show_teleop_debug = show_teleop_debug
        self.anchor_delay_s = anchor_delay_s
        self.anchor_ready_time = time.time() + anchor_delay_s
        self.grip_threshold = grip_threshold
        self.event_log_path = event_log_path
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
        self.last_joint_targets_deg = np.zeros(len(self.model.joint_names), dtype=float)
        self.last_packet_timestamp = None
        self.minimum_packet_timestamp = None
        self.last_debug_payload = None
        self._last_event_key = None
        self.quest_anchor_position = None
        self.quest_anchor_rotation = None
        self.sim_anchor_pose = None
        self._a_pressed_last = False

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
    ) -> tuple[np.ndarray, dict[str, Any]] | None:
        normalized = packet.get("normalized")
        if not isinstance(normalized, dict):
            return None
        hand = normalized.get("right_hand")
        if not isinstance(hand, dict):
            return None
        if not hand.get("enabled", True) or hand.get("clutch", False):
            return None
        if float(hand.get("grip", 0.0)) < self.grip_threshold:
            return None

        quest_position = np.asarray(hand["position"], dtype=float)
        quest_rotation = Rotation.from_quat(np.asarray(hand["orientation_xyzw"], dtype=float)).as_matrix()
        current_sim_pose = self.model.forward_kinematics(current_joint_targets_deg)

        if self.quest_anchor_position is None or self.sim_anchor_pose is None:
            if time.time() < self.anchor_ready_time:
                return None
            self.quest_anchor_position = quest_position.copy()
            self.quest_anchor_rotation = quest_rotation.copy()
            self.sim_anchor_pose = current_sim_pose.copy()
            self._append_event(
                {
                    "status": "anchor_captured",
                    "quest_anchor_position": self.quest_anchor_position.tolist(),
                    "sim_anchor_position": self.sim_anchor_pose[:3, 3].tolist(),
                },
                dedupe_key=("anchor_captured", tuple(np.round(self.quest_anchor_position, 6))),
            )

        quest_delta = quest_position - self.quest_anchor_position
        remapped_delta = quest_delta[list(self.quest_position_axes)] * self.quest_position_signs
        position_delta = self.world_rotation @ (remapped_delta * self.position_scale)
        quest_mapped_position = self.sim_anchor_pose[:3, 3] + position_delta
        target_position = quest_mapped_position + self.world_offset
        if self.position_only:
            target_rotation = self.sim_anchor_pose[:3, :3]
        else:
            relative_rotation = quest_rotation @ self.quest_anchor_rotation.T
            target_rotation = self.world_rotation @ relative_rotation @ self.sim_anchor_pose[:3, :3]
        self._update_teleop_debug_visuals(
            quest_anchor_position=self.quest_anchor_position,
            quest_current_position=quest_position,
            quest_mapped_position=quest_mapped_position,
            sim_target_position=target_position,
        )
        target_pose = self.kinematics_module.make_pose(position=target_position, rotation_matrix=target_rotation)
        return target_pose, hand


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
        visuals = [
            ("QuestMapped", quest_mapped_position, (1.0, 0.5, 0.0), 0.004),
            ("SimTarget", sim_target_position, (1.0, 0.0, 0.0), 0.005),
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

    def _get_stage(self):
        try:
            import omni.usd
        except ImportError:
            return None
        return omni.usd.get_context().get_stage()

    def _read_current_joint_targets_deg(self, stage) -> np.ndarray:
        joint_targets = []
        for joint_name in self.model.joint_names:
            prim = stage.GetPrimAtPath(f"{self.joint_root_path}/{joint_name}")
            if not prim.IsValid():
                raise RuntimeError(f"Joint prim not found: {self.joint_root_path}/{joint_name}")
            attr = prim.GetAttribute("drive:angular:physics:targetPosition")
            value = attr.Get() if attr.IsValid() else None
            joint_targets.append(float(value) if value is not None else 0.0)
        return np.asarray(joint_targets, dtype=float)

    def _write_joint_targets_deg(self, stage, joint_targets_deg: np.ndarray) -> None:
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
        self.quest_anchor_position = None
        self.quest_anchor_rotation = None
        self.sim_anchor_pose = None
        self.anchor_ready_time = time.time() + self.anchor_delay_s
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
        self.last_joint_targets_deg = target_values

    def solve_once(self, *, apply_to_stage: bool) -> dict[str, Any] | None:
        packet = self._load_latest_packet()
        if packet is None:
            return None

        packet_timestamp = packet.get("timestamp")
        normalized = packet.get("normalized") if isinstance(packet, dict) else None
        hand = normalized.get("right_hand") if isinstance(normalized, dict) else None
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
            current_joint_targets_deg = self._read_current_joint_targets_deg(stage)
        else:
            current_joint_targets_deg = self.last_joint_targets_deg

        target = self._packet_to_target_pose(packet, current_joint_targets_deg)
        if target is None:
            if self.quest_anchor_position is None and time.time() < self.anchor_ready_time:
                self._write_debug(
                    {
                        "status": "waiting_for_anchor",
                        "anchor_delay_s": self.anchor_delay_s,
                        "anchor_ready_time": self.anchor_ready_time,
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
                    "grip_threshold": self.grip_threshold,
                }
                self._append_event(
                    ignored_event,
                    dedupe_key=("ignored", ignored_event["reason"], round(float(grip or 0.0), 3), round(float(self.grip_threshold), 3)),
                )
                if self.last_debug_payload is None or self.last_debug_payload.get("status") != "applied":
                    self._write_debug({**ignored_event, "packet": packet})
            return None
        target_pose, hand_state = target

        solved_joint_targets_deg = self.model.inverse_kinematics(current_joint_targets_deg, target_pose)
        self.last_joint_targets_deg = solved_joint_targets_deg

        if stage is not None:
            self._write_joint_targets_deg(stage, solved_joint_targets_deg)

        result = {
            "timestamp": packet_timestamp,
            "joint_names": self.model.joint_names,
            "joint_targets_deg": solved_joint_targets_deg.tolist(),
            "target_position": target_pose[:3, 3].tolist(),
            "position_only": self.position_only,
            "quest_position_axes": list(self.quest_position_axes),
            "quest_position_signs": self.quest_position_signs.tolist(),
            "grip": float(hand_state.get("grip", 0.0)),
            "trigger": float(hand_state.get("trigger", 0.0)),
        }
        hand_position = np.asarray(hand_state.get("position", [0.0, 0.0, 0.0]), dtype=float)
        target_position = target_pose[:3, 3]
        event_payload = {
            "status": "applied" if stage is not None else "solved",
                "packet_timestamp": packet_timestamp,
                "position_only": self.position_only,
                "quest_position_axes": list(self.quest_position_axes),
                "quest_position_signs": self.quest_position_signs.tolist(),
                "raw_hand_position": hand_state.get("position"),
                "raw_hand_orientation_xyzw": hand_state.get("orientation_xyzw"),
                "quest_anchor_position": None if self.quest_anchor_position is None else self.quest_anchor_position.tolist(),
                "sim_anchor_position": None if self.sim_anchor_pose is None else self.sim_anchor_pose[:3, 3].tolist(),
                "quest_delta": None if self.quest_anchor_position is None else (hand_position - self.quest_anchor_position).tolist(),
                "mapped_delta": None if self.sim_anchor_pose is None else (target_position - self.sim_anchor_pose[:3, 3]).tolist(),
                "current_joint_targets_deg": current_joint_targets_deg.tolist(),
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

    def _on_update(self, _event: object) -> None:
        now = time.monotonic()
        if now - self._last_tick_time < self.interval_s:
            return
        self._last_tick_time = now
        try:
            result = self.controller.solve_once(apply_to_stage=self.apply_to_stage)
            debug_payload = self.controller.last_debug_payload or {}
            status = debug_payload.get("status")

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
