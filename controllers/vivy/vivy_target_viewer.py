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
TELEOP_STATE_PATH = Path("/home/viaan/huggingface/lerobot/src/lerobot/robots/vivy/fanout/teleop_state.py")
TELEOP_DEBUG_VISUALS_PATH = Path("/home/viaan/vivy_hopejr_sim/ui/teleop_debug_visuals.py")
STAGE_IO_PATH = Path("/home/viaan/vivy_hopejr_sim/controllers/stage_io.py")
VIVY_SIDE_PANEL_PATH = Path("/home/viaan/vivy_hopejr_sim/ui/vivy/vivy_side_panel.py")
VIVY_FLOW_PANEL_PATH = Path("/home/viaan/vivy_hopejr_sim/ui/vivy/vivy_flow_panel.py")
VIVY_FLOW_DETAIL_PANEL_PATH = Path("/home/viaan/vivy_hopejr_sim/ui/vivy/vivy_flow_detail_panel.py")
FLOW_CONTROL_PATH = Path("/tmp/vivy_flow_control.json")
STAGE_FEEDBACK_PATH = Path("/tmp/vivy_stage_feedback.json")
REAL_FEEDBACK_PATH = Path("/tmp/vivy_real_feedback.json")

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


def _load_json_file(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


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
        self._last_applied_feedback_timestamp: float | None = None
        self._last_applied_command_timestamp: float | None = None

        self.sim_config = _load_sim_config()
        controller_defaults = dict(self.sim_config.get("controller_defaults") or {})
        self.end_effector_path = controller_defaults.get("end_effector_path", "/World/JointTest/PalmBody/EndEffector")
        self.teleop_debug_root = controller_defaults.get("teleop_debug_root", "/World/JointTest/TeleopDebug")
        self.articulation_root_path = controller_defaults.get("articulation_root_path", "/World/JointTest")
        self.joint_root_path = controller_defaults.get("joint_root_path", "/World/JointTest/Joints")
        self.world_offset = np.asarray(controller_defaults.get("world_offset", [0.0, 0.0, 0.0]), dtype=float)
        self.real_feedback_max_age_s = float(controller_defaults.get("packet_stale_timeout_s", 0.75))
        joint_names = list(self.sim_config["joint_names"])
        self.joint_names = list(joint_names)
        neutral_deg = np.asarray([float(self.sim_config["joints"][name]["neutral_deg"]) for name in joint_names], dtype=float)
        self.kinematics = VivyArmKinematics.from_json(SIM_CONFIG_PATH)
        self.model_neutral_pose = self.kinematics.forward_kinematics(neutral_deg)
        self.model_to_stage_transform = np.eye(4)
        self._anchor_model_to_stage_transform = np.eye(4)
        self._stage_anchor_pose = None
        self._last_waiting_for_anchor = True
        HopeJrStageIo = _load_stage_io_class()
        TeleopDebugVisuals = _load_visuals_class()
        VivySidePanel = _load_side_panel_class()
        VivyFlowPanel = _load_flow_panel_class()
        VivyFlowDetailPanel = _load_flow_detail_panel_class()
        self.stage_io = HopeJrStageIo(
            articulation_root_path=self.articulation_root_path,
            joint_root_path=self.joint_root_path,
            end_effector_path=self.end_effector_path,
            joint_names=joint_names,
        )
        self.visuals = TeleopDebugVisuals(teleop_debug_root=self.teleop_debug_root, enabled=True)
        self.side_panel = VivySidePanel()
        self.flow_panel = VivyFlowPanel()
        self.flow_detail_panel = VivyFlowDetailPanel()
        self._joint_write_ready = False
        self._last_joint_write_warn_time = 0.0
        self._last_panel_signal_timestamp: float | None = None
        self._last_panel_signal_arrival_time: float | None = None
        self._last_panel_real_feedback_timestamp: float | None = None
        self._last_panel_real_feedback_arrival_time: float | None = None

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
            else:
                merged_row["actual_raw"] = "-"
                merged_row["actual_deg"] = "-"
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
        hz = None
        if self._last_panel_real_feedback_timestamp is not None and self._last_panel_real_feedback_arrival_time is not None:
            dt = timestamp_f - self._last_panel_real_feedback_timestamp
            if dt > 1e-6:
                hz = 1.0 / dt
        self._last_panel_real_feedback_timestamp = timestamp_f
        self._last_panel_real_feedback_arrival_time = arrival_now
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
        try:
            self.stage_io.write_joint_targets_deg(stage, joint_positions_deg)
            if update_state:
                self.stage_io.write_joint_state_deg(stage, joint_positions_deg)
            self._joint_write_ready = True
        except Exception as exc:
            exc_text = str(exc)
            now = time.monotonic()
            if "Articulation unavailable" in exc_text:
                self._joint_write_ready = False
            elif now - self._last_joint_write_warn_time > 2.0:
                print(f"Vivy target viewer joint write failed: {exc}")
                self._last_joint_write_warn_time = now

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
            self.side_panel.update(panel_payload)
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

        stage_joint_positions_deg = self.stage_io.read_stage_joint_positions_deg(stage)
        if stage_joint_positions_deg is not None:
            try:
                _write_stage_feedback(
                    STAGE_FEEDBACK_PATH,
                    {
                        "timestamp": time.time(),
                        "joint_names": list(self.joint_names),
                        **{
                            f"{joint_name}.pos_deg": float(stage_joint_positions_deg[index])
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
        self._last_waiting_for_anchor = waiting_for_anchor

        signal_timestamp = self._coerce_timestamp(payload)
        feedback_timestamp = self._coerce_timestamp(real_feedback)
        teleop_changed = signal_timestamp is not None and signal_timestamp != self._last_signal_timestamp
        feedback_changed = feedback_timestamp is not None and feedback_timestamp != self._last_applied_feedback_timestamp

        joint_targets_deg = payload.get("current_joint_targets_deg")
        if real_joint_positions_deg is not None:
            if feedback_changed:
                self._write_sim_joint_positions(stage, np.asarray(real_joint_positions_deg, dtype=float), update_state=True)
                self._last_applied_feedback_timestamp = feedback_timestamp
        elif (
            teleop_changed
            and isinstance(joint_targets_deg, list)
            and len(joint_targets_deg) == len(self.sim_config["joint_names"])
        ):
            self._write_sim_joint_positions(stage, np.asarray(joint_targets_deg, dtype=float), update_state=False)
            self._last_applied_command_timestamp = signal_timestamp

        if teleop_changed:
            self._last_signal_timestamp = signal_timestamp
        else:
            return

        target_pose_model = payload.get("target_pose_model")
        if target_pose_model is None:
            target_pose_stage = stage_pose
        else:
            target_pose_model = np.asarray(target_pose_model, dtype=float)
            if self._stage_anchor_pose is not None and np.allclose(
                np.asarray(payload.get("conditioned_position_delta_world") or [0.0, 0.0, 0.0], dtype=float),
                0.0,
                atol=1e-6,
            ):
                try:
                    self._anchor_model_to_stage_transform = self._stage_anchor_pose @ np.linalg.inv(target_pose_model)
                except np.linalg.LinAlgError:
                    self._anchor_model_to_stage_transform = self.model_to_stage_transform
            target_pose_stage = self._anchor_model_to_stage_transform @ target_pose_model

        sim_target_position = np.asarray(target_pose_stage[:3, 3], dtype=float)
        show_pitch_frames = bool(flow_control.get("show_pitch_frames", False))
        pitch_visual = self._build_pitch_visual(stage) if show_pitch_frames else None
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
