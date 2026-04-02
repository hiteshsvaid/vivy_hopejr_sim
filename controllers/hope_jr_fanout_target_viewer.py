from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np


SIM_CONFIG_PATH = Path("/home/viaan/huggingface/lerobot/src/lerobot/robots/hope_jr/hope_global_config.json")
KINEMATICS_PATH = Path("/home/viaan/huggingface/lerobot/src/lerobot/robots/hope_jr/hope_jr_arm_kinematics.py")
SHARED_SIGNAL_PATH = Path("/home/viaan/huggingface/lerobot/src/lerobot/robots/hope_jr/fanout/shared_target_signal.py")
TELEOP_DEBUG_VISUALS_PATH = Path("/home/viaan/vivy_hopejr_sim/ui/teleop_debug_visuals.py")

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
    module = _load_module("hope_jr_fanout_kinematics", KINEMATICS_PATH)
    return module.HopeJrArmKinematics


def _load_shared_signal_helpers():
    module = _load_module("hope_jr_fanout_shared_signal", SHARED_SIGNAL_PATH)
    return module.DEFAULT_SHARED_TARGET_SIGNAL_PATH, module.read_shared_target_signal


def _load_visuals_class():
    module = _load_module("hope_jr_fanout_visuals", TELEOP_DEBUG_VISUALS_PATH)
    return module.TeleopDebugVisuals


class HopeJrFanoutTargetViewer:
    def __init__(self, *, signal_path: str | Path | None = None, interval_s: float = 0.05):
        default_signal_path, read_shared_target_signal = _load_shared_signal_helpers()
        HopeJrArmKinematics = _load_kinematics_class()
        self._read_shared_target_signal = read_shared_target_signal
        self.signal_path = Path(default_signal_path if signal_path is None else signal_path)
        self.interval_s = float(interval_s)
        self._last_tick_time = 0.0
        self._subscription = None
        self._last_signal_timestamp = None

        self.sim_config = _load_sim_config()
        controller_defaults = dict(self.sim_config.get("controller_defaults") or {})
        self.end_effector_path = controller_defaults.get("end_effector_path", "/World/JointTest/PalmBody/EndEffector")
        self.teleop_debug_root = controller_defaults.get("teleop_debug_root", "/World/JointTest/TeleopDebug")
        joint_names = list(self.sim_config["joint_names"])
        neutral_deg = np.asarray([float(self.sim_config["joints"][name]["neutral_deg"]) for name in joint_names], dtype=float)
        self.kinematics = HopeJrArmKinematics.from_json(SIM_CONFIG_PATH)
        self.model_neutral_pose = self.kinematics.forward_kinematics(neutral_deg)
        self.model_to_stage_transform = np.eye(4)
        TeleopDebugVisuals = _load_visuals_class()
        self.visuals = TeleopDebugVisuals(teleop_debug_root=self.teleop_debug_root, enabled=True)

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

    def _on_update(self, _event: object) -> None:
        now = time.monotonic()
        if now - self._last_tick_time < self.interval_s:
            return
        self._last_tick_time = now

        stage = self._read_stage()
        stage_pose = self._maybe_refresh_transform(stage)
        if stage is None or stage_pose is None:
            return

        payload = self._read_shared_target_signal(self.signal_path)
        if not isinstance(payload, dict):
            return

        signal_timestamp = payload.get("timestamp")
        if signal_timestamp == self._last_signal_timestamp:
            return
        self._last_signal_timestamp = signal_timestamp

        target_pose_model = payload.get("target_pose_model")
        waiting_for_anchor = bool(payload.get("waiting_for_anchor", True))
        if target_pose_model is None:
            target_pose_stage = stage_pose
        else:
            target_pose_stage = self.model_to_stage_transform @ np.asarray(target_pose_model, dtype=float)

        sim_target_position = np.asarray(target_pose_stage[:3, 3], dtype=float)
        self.visuals.update(
            stage,
            quest_anchor_position=np.asarray(payload.get("quest_anchor_position") or [0.0, 0.0, 0.0], dtype=float),
            quest_current_position=np.asarray((payload.get("hand_state") or {}).get("position") or [0.0, 0.0, 0.0], dtype=float),
            quest_mapped_position=sim_target_position,
            sim_target_position=sim_target_position,
            reference_position=stage_pose[:3, 3],
            actual_end_effector_position=stage_pose[:3, 3],
            actual_end_effector_pose=stage_pose,
            waiting_for_anchor=waiting_for_anchor,
        )

    def start(self):
        import omni.kit.app

        app = omni.kit.app.get_app()
        self._subscription = app.get_update_event_stream().create_subscription_to_pop(
            self._on_update,
            name="HopeJrFanoutTargetViewer",
        )
        print(f"Hope Jr fan-out target viewer subscribed at {self.interval_s:.3f}s interval")
        return self

    def stop(self) -> None:
        self._subscription = None
        print("Hope Jr fan-out target viewer unsubscribed")


def start_script_editor_loop(*, signal_path: str | Path | None = None, interval_s: float = 0.05):
    global _ACTIVE_LOOP
    stop_script_editor_loop()
    _ACTIVE_LOOP = HopeJrFanoutTargetViewer(signal_path=signal_path, interval_s=interval_s).start()
    return _ACTIVE_LOOP


def stop_script_editor_loop() -> None:
    global _ACTIVE_LOOP
    if _ACTIVE_LOOP is not None:
        _ACTIVE_LOOP.stop()
        _ACTIVE_LOOP = None
