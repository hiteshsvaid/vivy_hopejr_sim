#!/usr/bin/env python3

import argparse
import importlib.util
import json
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
    ):
        self.lerobot_repo = lerobot_repo
        self.packet_path = packet_path
        self.joint_root_path = joint_root_path.rstrip("/")
        self.position_scale = position_scale
        self.world_offset = world_offset
        self.world_rotation = Rotation.from_euler("XYZ", world_rotate_xyz_deg, degrees=True).as_matrix()
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

    def _load_kinematics_module(self, module_path: Path):
        spec = importlib.util.spec_from_file_location("hope_jr_arm_kinematics", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load Hope Jr kinematics module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def _load_latest_packet(self) -> dict[str, Any] | None:
        if not self.packet_path.is_file():
            return None
        return json.loads(self.packet_path.read_text())

    def _packet_to_target_pose(self, packet: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]] | None:
        normalized = packet.get("normalized")
        if not isinstance(normalized, dict):
            return None
        hand = normalized.get("right_hand")
        if not isinstance(hand, dict):
            return None
        if not hand.get("enabled", True) or hand.get("clutch", False):
            return None

        position = np.asarray(hand["position"], dtype=float) * self.position_scale
        quest_rotation = Rotation.from_quat(np.asarray(hand["orientation_xyzw"], dtype=float)).as_matrix()
        target_position = self.world_rotation @ position + self.world_offset
        target_rotation = self.world_rotation @ quest_rotation
        target_pose = self.kinematics_module.make_pose(position=target_position, rotation_matrix=target_rotation)
        return target_pose, hand

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

    def solve_once(self, *, apply_to_stage: bool) -> dict[str, Any] | None:
        packet = self._load_latest_packet()
        if packet is None:
            return None

        packet_timestamp = packet.get("timestamp")
        if packet_timestamp is None:
            return None
        if self.minimum_packet_timestamp is not None and packet_timestamp <= self.minimum_packet_timestamp:
            return None
        if packet_timestamp == self.last_packet_timestamp:
            return None
        self.last_packet_timestamp = packet_timestamp

        target = self._packet_to_target_pose(packet)
        if target is None:
            return None
        target_pose, hand_state = target

        stage = self._get_stage() if apply_to_stage else None
        if stage is not None:
            current_joint_targets_deg = self._read_current_joint_targets_deg(stage)
        else:
            current_joint_targets_deg = self.last_joint_targets_deg

        solved_joint_targets_deg = self.model.inverse_kinematics(current_joint_targets_deg, target_pose)
        self.last_joint_targets_deg = solved_joint_targets_deg

        if stage is not None:
            self._write_joint_targets_deg(stage, solved_joint_targets_deg)

        return {
            "timestamp": packet_timestamp,
            "joint_names": self.model.joint_names,
            "joint_targets_deg": solved_joint_targets_deg.tolist(),
            "target_position": target_pose[:3, 3].tolist(),
            "grip": float(hand_state.get("grip", 0.0)),
            "trigger": float(hand_state.get("trigger", 0.0)),
        }


class HopeJrIsaacUpdateLoop:
    def __init__(self, controller: HopeJrSimIkController, *, apply_to_stage: bool, interval_s: float):
        self.controller = controller
        self.apply_to_stage = apply_to_stage
        self.interval_s = interval_s
        self._subscription = None
        self._last_tick_time = 0.0

    def _on_update(self, _event: object) -> None:
        now = time.monotonic()
        if now - self._last_tick_time < self.interval_s:
            return
        self._last_tick_time = now
        try:
            result = self.controller.solve_once(apply_to_stage=self.apply_to_stage)
            if result is not None:
                print(json.dumps(result, indent=2))
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
        print("Hope Jr IK controller unsubscribed from Isaac update stream")


def build_controller_from_args(args: argparse.Namespace) -> HopeJrSimIkController:
    return HopeJrSimIkController(
        lerobot_repo=args.lerobot_repo,
        packet_path=args.packet_path,
        joint_root_path=args.joint_root_path,
        position_scale=args.position_scale,
        world_offset=np.asarray(args.world_offset, dtype=float),
        world_rotate_xyz_deg=np.asarray(args.world_rotate_xyz, dtype=float),
    )


def start_script_editor_loop(
    *,
    lerobot_repo: str | Path = DEFAULT_LEROBOT_REPO,
    packet_path: str | Path = DEFAULT_PACKET_PATH,
    joint_root_path: str = DEFAULT_JOINT_ROOT_PATH,
    position_scale: float = 1.0,
    world_offset: list[float] | tuple[float, float, float] = (0.0, 0.0, 0.0),
    world_rotate_xyz: list[float] | tuple[float, float, float] = (0.0, 0.0, 0.0),
    interval_s: float = 0.05,
    dry_run: bool = False,
    consume_only_new: bool = True,
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
    )
    if consume_only_new:
        controller.minimum_packet_timestamp = time.time()
    _ACTIVE_LOOP = HopeJrIsaacUpdateLoop(controller, apply_to_stage=not dry_run, interval_s=interval_s).start()
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
