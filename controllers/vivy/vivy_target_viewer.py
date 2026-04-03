from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np


SIM_CONFIG_PATH = Path("/home/viaan/huggingface/lerobot/src/lerobot/robots/vivy/vivy_global_config.json")
KINEMATICS_PATH = Path("/home/viaan/huggingface/lerobot/src/lerobot/robots/vivy/vivy_arm_kinematics.py")
TELEOP_STATE_PATH = Path("/home/viaan/huggingface/lerobot/src/lerobot/robots/vivy/fanout/teleop_state.py")
TELEOP_DEBUG_VISUALS_PATH = Path("/home/viaan/vivy_hopejr_sim/ui/teleop_debug_visuals.py")
VIVY_SIDE_PANEL_PATH = Path("/home/viaan/vivy_hopejr_sim/ui/vivy/vivy_side_panel.py")
VIVY_FLOW_PANEL_PATH = Path("/home/viaan/vivy_hopejr_sim/ui/vivy/vivy_flow_panel.py")
VIVY_FLOW_DETAIL_PANEL_PATH = Path("/home/viaan/vivy_hopejr_sim/ui/vivy/vivy_flow_detail_panel.py")
FLOW_CONTROL_PATH = Path("/tmp/vivy_flow_control.json")

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


def _load_shared_signal_helpers():
    module = _load_module("vivy_target_viewer_teleop_state", TELEOP_STATE_PATH)
    return module.DEFAULT_TELEOP_STATE_PATH, module.read_teleop_state


def _load_visuals_class():
    module = _load_module("vivy_target_viewer_visuals", TELEOP_DEBUG_VISUALS_PATH)
    return module.TeleopDebugVisuals


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


class VivyTargetViewer:
    def __init__(self, *, signal_path: str | Path | None = None, interval_s: float = 0.05):
        default_signal_path, read_teleop_state = _load_shared_signal_helpers()
        VivyArmKinematics = _load_kinematics_class()
        self._read_teleop_state = read_teleop_state
        self.signal_path = Path(default_signal_path if signal_path is None else signal_path)
        self.interval_s = float(interval_s)
        self._last_tick_time = 0.0
        self._subscription = None
        self._last_signal_timestamp = None

        self.sim_config = _load_sim_config()
        controller_defaults = dict(self.sim_config.get("controller_defaults") or {})
        self.end_effector_path = controller_defaults.get("end_effector_path", "/World/JointTest/PalmBody/EndEffector")
        self.teleop_debug_root = controller_defaults.get("teleop_debug_root", "/World/JointTest/TeleopDebug")
        self.world_offset = np.asarray(controller_defaults.get("world_offset", [0.0, 0.0, 0.0]), dtype=float)
        joint_names = list(self.sim_config["joint_names"])
        neutral_deg = np.asarray([float(self.sim_config["joints"][name]["neutral_deg"]) for name in joint_names], dtype=float)
        self.kinematics = VivyArmKinematics.from_json(SIM_CONFIG_PATH)
        self.model_neutral_pose = self.kinematics.forward_kinematics(neutral_deg)
        self.model_to_stage_transform = np.eye(4)
        self._stage_anchor_pose = None
        self._last_waiting_for_anchor = True
        TeleopDebugVisuals = _load_visuals_class()
        VivySidePanel = _load_side_panel_class()
        VivyFlowPanel = _load_flow_panel_class()
        VivyFlowDetailPanel = _load_flow_detail_panel_class()
        self.visuals = TeleopDebugVisuals(teleop_debug_root=self.teleop_debug_root, enabled=True)
        self.side_panel = VivySidePanel()
        self.flow_panel = VivyFlowPanel()
        self.flow_detail_panel = VivyFlowDetailPanel()

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

        payload = self._read_teleop_state(self.signal_path)
        if not isinstance(payload, dict):
            return
        flow_control = _read_flow_control()
        try:
            self.side_panel.update(payload)
        except Exception:
            pass
        try:
            self.flow_panel.update(payload, flow_control)
        except Exception:
            pass
        try:
            self.flow_detail_panel.update(payload, flow_control)
        except Exception:
            pass

        waiting_for_anchor = bool(payload.get("waiting_for_anchor", True))
        if waiting_for_anchor:
            self._stage_anchor_pose = None
        elif self._stage_anchor_pose is None or self._last_waiting_for_anchor:
            self._stage_anchor_pose = stage_pose.copy()
        self._last_waiting_for_anchor = waiting_for_anchor

        signal_timestamp = payload.get("timestamp")
        if signal_timestamp == self._last_signal_timestamp:
            return
        self._last_signal_timestamp = signal_timestamp

        target_pose_model = payload.get("target_pose_model")
        if target_pose_model is None:
            target_pose_stage = stage_pose
        elif self._stage_anchor_pose is not None and payload.get("position_delta_world") is not None:
            target_pose_stage = self._stage_anchor_pose.copy()
            target_pose_stage[:3, 3] = (
                self._stage_anchor_pose[:3, 3]
                + np.asarray(payload.get("position_delta_world"), dtype=float)
                + self.world_offset
            )
        else:
            target_pose_stage = self.model_to_stage_transform @ np.asarray(target_pose_model, dtype=float)

        sim_target_position = np.asarray(target_pose_stage[:3, 3], dtype=float)
        self.visuals.enabled = bool(flow_control.get("sim_view_enabled", True))
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
