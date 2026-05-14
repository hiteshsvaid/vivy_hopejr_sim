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


def _load_head_teleop_mapper_class():
    module = _load_module("vivy_target_viewer_head_teleop_mapper", Path("/home/viaan/vivy_hopejr_sim/controllers/head_teleop_mapper.py"))
    return module.HeadTeleopMapper


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


def _parse_rotation_axis(value: object) -> int:
    axis = str(value).strip().lower()
    axis_map = {"x": 0, "y": 1, "z": 2}
    if axis not in axis_map:
        raise ValueError(f"unsupported rotation axis: {value!r}")
    return axis_map[axis]


def _parse_button_or_axis_input_source(value: object) -> str:
    source = str(value or "none").strip().lower()
    alias_map = {
        "a": "a_pressed",
        "b": "b_pressed",
        "x": "x_pressed",
        "y": "y_pressed",
        "primary_button": "a_pressed",
        "secondary_button": "b_pressed",
    }
    return alias_map.get(source, source)


def _read_direct_input_source(hand_state: dict[str, object], source: str) -> float:
    if source == "none":
        return 0.0
    if source in {"grip", "trigger"}:
        return max(0.0, float(hand_state.get(source, 0.0)))
    return 1.0 if bool(hand_state.get(source, False)) else 0.0


class VivyTargetViewer:
    def __init__(self, *, signal_path: str | Path | None = None, interval_s: float = 0.05):
        default_signal_path, read_teleop_state = _load_shared_signal_helpers()
        VivyArmKinematics = _load_kinematics_class()
        make_pose = _load_kinematics_make_pose()
        QuestTeleopMapper = _load_quest_teleop_mapper_class()
        HeadTeleopMapper = _load_head_teleop_mapper_class()
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
        self._last_left_log_signature = None

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
        left_ik_joint_names = list(left_chain_config.get("joint_names") or [])
        if not left_ik_joint_names:
            left_ik_joint_names = list(self.ik_joint_names)
        left_controlled_joint_names = list(left_chain_config.get("controlled_joint_names") or left_ik_joint_names)
        self.left_ik_joint_names = list(left_ik_joint_names)
        self.left_controlled_joint_names = list(left_controlled_joint_names)
        self.left_controlled_to_ik_indices = np.asarray(
            [left_controlled_joint_names.index(name) for name in left_ik_joint_names],
            dtype=np.int64,
        )
        self.left_kinematics = VivyArmKinematics.from_json(SIM_CONFIG_PATH, chain_name="left_arm")
        self.left_position_scale = float(controller_defaults.get("position_scale", 1.0))
        self.left_world_offset = np.asarray(controller_defaults.get("world_offset", [0.0, 0.0, 0.0]), dtype=float)
        self.left_world_rotate_xyz = np.asarray(controller_defaults.get("world_rotate_xyz", [0.0, 0.0, 0.0]), dtype=float)
        self.left_quest_position_axes = _parse_position_axes(
            str(controller_defaults.get("left_quest_position_axes", controller_defaults.get("quest_position_axes", "xyz")))
        )
        self.left_quest_position_signs = np.asarray(
            controller_defaults.get("left_quest_position_signs", [1.0, 1.0, 1.0]),
            dtype=float,
        )
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
        self.left_ik_damping = float(controller_defaults.get("ik_damping", 0.02))
        self.left_ik_max_iteration = int(controller_defaults.get("ik_max_iteration", 80))
        self.left_ik_max_step_deg = float(controller_defaults.get("ik_max_step_deg", 2.0))
        self.left_ik_jacobian_mode = str(controller_defaults.get("ik_jacobian_mode", "analytic"))
        self.left_neutral_joint_positions_deg = np.asarray(
            [float(self.sim_config["joints"][name]["neutral_deg"]) for name in left_ik_joint_names],
            dtype=float,
        )
        self.left_joint_weights = np.asarray(
            [float(self.sim_config["joints"][name].get("weight", 1.0)) for name in left_ik_joint_names],
            dtype=float,
        )
        self.left_active_joint_mask = np.asarray(
            [0.0 if bool(self.sim_config["joints"][name].get("hold_start", False)) else 1.0 for name in left_ik_joint_names],
            dtype=float,
        )

        head_joint_names = ("head_pan", "head_tilt")
        head_neutral_joint_targets_deg = np.asarray(
            [float(self.sim_config["joints"][name]["neutral_deg"]) for name in head_joint_names],
            dtype=float,
        )
        head_lower_joint_limits_deg = np.asarray(
            [float(self.sim_config["joints"][name]["min_deg"]) for name in head_joint_names],
            dtype=float,
        )
        head_upper_joint_limits_deg = np.asarray(
            [float(self.sim_config["joints"][name]["max_deg"]) for name in head_joint_names],
            dtype=float,
        )
        head_max_delta_deg_per_tick = np.asarray(
            [float(self.sim_config["joints"][name].get("output_max_delta_deg_per_tick", controller_defaults.get("output_max_delta_deg_per_tick", 2.0))) for name in head_joint_names],
            dtype=float,
        )
        self.head_teleop_mapper = HeadTeleopMapper(
            head_joint_names=head_joint_names,
            neutral_joint_targets_deg=head_neutral_joint_targets_deg,
            lower_joint_limits_deg=head_lower_joint_limits_deg,
            upper_joint_limits_deg=head_upper_joint_limits_deg,
            pan_input_clamp_deg=float(controller_defaults.get("head_pan_input_clamp_deg", 60.0)),
            tilt_input_clamp_deg=float(controller_defaults.get("head_tilt_input_clamp_deg", 30.0)),
            max_delta_deg_per_tick=head_max_delta_deg_per_tick,
        )
        self.head_stage_io = HopeJrStageIo(
            articulation_root_path=self.articulation_root_path,
            joint_root_path=self.joint_root_path,
            end_effector_path=self.end_effector_path,
            joint_names=list(head_joint_names),
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
            joint_names=left_controlled_joint_names,
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

    def _log_left_mapping(self, left_map_result, left_hand: dict) -> None:
        current = getattr(left_map_result, "quest_current_position", None)
        anchor = getattr(left_map_result, "quest_anchor_position", None)
        target = getattr(left_map_result, "sim_target_position_stage", None)
        signature = (
            bool(getattr(left_map_result, "tracked", False)) if left_map_result is not None else False,
            bool(getattr(left_map_result, "waiting_for_anchor", False)) if left_map_result is not None else True,
            bool(getattr(left_map_result, "follow_target_enabled", False)) if left_map_result is not None else False,
            tuple(np.round(np.asarray(current, dtype=float), 3)) if current is not None else None,
            tuple(np.round(np.asarray(anchor, dtype=float), 3)) if anchor is not None else None,
            tuple(np.round(np.asarray(target, dtype=float), 3)) if target is not None else None,
            bool(left_map_result is not None and getattr(left_map_result, "anchor_captured_payload", None) is not None),
            bool(left_hand.get("thumbstick_click", False)) if isinstance(left_hand, dict) else False,
        )
        if signature == self._last_left_log_signature:
            return
        self._last_left_log_signature = signature
        current_text = "n/a" if current is None else np.array2string(np.asarray(current, dtype=float), precision=2, separator=", ")
        anchor_text = "n/a" if anchor is None else np.array2string(np.asarray(anchor, dtype=float), precision=2, separator=", ")
        target_text = "n/a" if target is None else np.array2string(np.asarray(target, dtype=float), precision=2, separator=", ")
        if left_map_result is None:
            print(f"Vivy left: waiting_left_thumbstick_click current={current_text} anchor={anchor_text} target={target_text}")
            return
        state = "tracked" if bool(getattr(left_map_result, "tracked", False)) else "untracked"
        if bool(getattr(left_map_result, "follow_target_enabled", False)):
            state = f"{state}/armed"
        elif bool(getattr(left_map_result, "waiting_for_anchor", False)):
            state = f"{state}/waiting_left_thumbstick_click"
        if bool(getattr(left_map_result, "tracking_lost", False)):
            state = f"{state}/lost"
        print(f"Vivy left: {state} current={current_text} anchor={anchor_text} target={target_text}")
        anchor_payload = getattr(left_map_result, "anchor_captured_payload", None)
        if isinstance(anchor_payload, dict):
            print(
                "Vivy left: left-thumbstick armed anchor captured "
                f"position={anchor_payload.get('quest_anchor_position')} target={anchor_payload.get('anchor_joint_targets_deg')}"
            )

    def _solve_left_arm_targets(
        self,
        *,
        current_joint_targets_deg: np.ndarray,
        target_pose_model: np.ndarray,
    ) -> np.ndarray:
        return self.left_kinematics.inverse_kinematics(
            np.asarray(current_joint_targets_deg, dtype=float),
            np.asarray(target_pose_model, dtype=float),
            ik_max_iteration=self.left_ik_max_iteration,
            damping=self.left_ik_damping,
            max_step_deg=self.left_ik_max_step_deg,
            orientation_weight=0.0,
            active_joint_mask=self.left_active_joint_mask,
            joint_weights=self.left_joint_weights,
            neutral_positions_deg=self.left_neutral_joint_positions_deg,
            jacobian_mode=self.left_ik_jacobian_mode,
        )

    def _merge_left_ik_joint_targets_deg(
        self,
        current_controlled_targets_deg: np.ndarray,
        proposed_ik_targets_deg: np.ndarray,
    ) -> np.ndarray:
        merged = np.asarray(current_controlled_targets_deg, dtype=float).copy()
        merged[self.left_controlled_to_ik_indices] = np.asarray(proposed_ik_targets_deg, dtype=float)
        return merged

    def _apply_rotation_direct_overrides(
        self,
        *,
        joint_names: list[str],
        joint_targets_deg: np.ndarray,
        hand_state: dict[str, object] | None,
        quest_anchor_rotation: np.ndarray | None,
    ) -> np.ndarray:
        result = np.asarray(joint_targets_deg, dtype=float).copy()
        if not isinstance(hand_state, dict) or quest_anchor_rotation is None:
            return result
        orientation_xyzw = hand_state.get("orientation_xyzw")
        if orientation_xyzw is None:
            return result
        try:
            current_rotation = Rotation.from_quat(np.asarray(orientation_xyzw, dtype=float)).as_matrix()
            relative_rotation = current_rotation @ np.asarray(quest_anchor_rotation, dtype=float).T
            relative_rotvec_deg = Rotation.from_matrix(relative_rotation).as_rotvec(degrees=True)
        except Exception:
            return result
        joints = dict(self.sim_config.get("joints") or {})
        for index, joint_name in enumerate(joint_names):
            joint_entry = dict(joints.get(joint_name) or {})
            direct_input = dict(joint_entry.get("direct_input") or {})
            if str(direct_input.get("source", "none")).strip().lower() != "rotation":
                continue
            try:
                axis_index = _parse_rotation_axis(direct_input.get("axis", "z"))
                sign = float(direct_input.get("sign", 1.0))
                lower = float(joint_entry.get("min_deg", result[index]))
                upper = float(joint_entry.get("max_deg", result[index]))
            except (TypeError, ValueError):
                continue
            result[index] = float(np.clip(relative_rotvec_deg[axis_index] * sign, lower, upper))
        return result

    def _apply_thumbstick_direct_overrides(
        self,
        *,
        joint_names: list[str],
        joint_targets_deg: np.ndarray,
        hand_state: dict[str, object] | None,
    ) -> np.ndarray:
        result = np.asarray(joint_targets_deg, dtype=float).copy()
        if not isinstance(hand_state, dict):
            return result
        thumbstick = np.asarray(hand_state.get("thumbstick", [0.0, 0.0]), dtype=float)
        joints = dict(self.sim_config.get("joints") or {})
        axis_map = {"x": 0, "y": 1}
        for index, joint_name in enumerate(joint_names):
            joint_entry = dict(joints.get(joint_name) or {})
            direct_input = dict(joint_entry.get("direct_input") or {})
            if str(direct_input.get("source", "none")).strip().lower() != "thumbstick":
                continue
            try:
                axis_name = str(direct_input.get("axis", "x")).strip().lower()
                axis_index = axis_map[axis_name]
                sign = float(direct_input.get("sign", 1.0))
                deadband = float(direct_input.get("deadband", 0.1))
                lower = float(joint_entry.get("min_deg", result[index]))
                upper = float(joint_entry.get("max_deg", result[index]))
            except (KeyError, TypeError, ValueError):
                continue
            thumbstick_value = float(thumbstick[axis_index]) if thumbstick.size > axis_index else 0.0
            if abs(thumbstick_value) < deadband:
                continue
            joint_tick_deg = float(joint_entry.get("output_max_delta_deg_per_tick", 0.0))
            result[index] = float(np.clip(result[index] + thumbstick_value * sign * joint_tick_deg, lower, upper))
        return result

    def _apply_button_pair_direct_overrides(
        self,
        *,
        joint_names: list[str],
        joint_targets_deg: np.ndarray,
        hand_state: dict[str, object] | None,
    ) -> np.ndarray:
        result = np.asarray(joint_targets_deg, dtype=float).copy()
        if not isinstance(hand_state, dict):
            return result
        joints = dict(self.sim_config.get("joints") or {})
        for index, joint_name in enumerate(joint_names):
            joint_entry = dict(joints.get(joint_name) or {})
            direct_input = dict(joint_entry.get("direct_input") or {})
            if str(direct_input.get("source", "none")).strip().lower() != "button_pair":
                continue
            inward_source = _parse_button_or_axis_input_source(direct_input.get("inward_source", "none"))
            outward_source = _parse_button_or_axis_input_source(direct_input.get("outward_source", "none"))
            try:
                sign = float(direct_input.get("sign", 1.0))
                lower = float(joint_entry.get("min_deg", result[index]))
                upper = float(joint_entry.get("max_deg", result[index]))
            except (TypeError, ValueError):
                continue
            drive = sign * (
                _read_direct_input_source(hand_state, outward_source)
                - _read_direct_input_source(hand_state, inward_source)
            )
            if abs(drive) <= 1e-6:
                continue
            joint_tick_deg = float(joint_entry.get("output_max_delta_deg_per_tick", 0.0))
            result[index] = float(np.clip(result[index] + drive * joint_tick_deg, lower, upper))
        return result

    def _log_head_mapping(self, head_map_result, head_state: dict) -> None:
        current = getattr(head_map_result, "head_current_degrees", None)
        anchor = getattr(head_map_result, "head_anchor_degrees", None)
        target = getattr(head_map_result, "target_joint_targets_deg", None)
        signature = (
            bool(getattr(head_map_result, "tracked", False)) if head_map_result is not None else False,
            bool(getattr(head_map_result, "waiting_for_anchor", False)) if head_map_result is not None else True,
            bool(getattr(head_map_result, "follow_target_enabled", False)) if head_map_result is not None else False,
            tuple(np.round(np.asarray(current, dtype=float), 3)) if current is not None else None,
            tuple(np.round(np.asarray(anchor, dtype=float), 3)) if anchor is not None else None,
            tuple(np.round(np.asarray(target, dtype=float), 3)) if target is not None else None,
            bool(head_map_result is not None and getattr(head_map_result, "anchor_captured_payload", None) is not None),
            bool(head_state.get("is_tracked", False)) if isinstance(head_state, dict) else False,
        )
        if signature == getattr(self, "_last_head_log_signature", None):
            return
        self._last_head_log_signature = signature
        current_text = "n/a" if current is None else np.array2string(np.asarray(current, dtype=float), precision=2, separator=", ")
        anchor_text = "n/a" if anchor is None else np.array2string(np.asarray(anchor, dtype=float), precision=2, separator=", ")
        target_text = "n/a" if target is None else np.array2string(np.asarray(target, dtype=float), precision=2, separator=", ")
        if head_map_result is None:
            print(f"Vivy head: waiting_right_thumbclick current={current_text} anchor={anchor_text} target={target_text}")
            return
        state = "tracked" if bool(getattr(head_map_result, "tracked", False)) else "untracked"
        if bool(getattr(head_map_result, "follow_target_enabled", False)):
            state = f"{state}/armed"
        elif bool(getattr(head_map_result, "waiting_for_anchor", False)):
            state = f"{state}/waiting_right_thumbclick"
        if bool(getattr(head_map_result, "tracking_lost", False)):
            state = f"{state}/lost"
        print(f"Vivy head: {state} current={current_text} anchor={anchor_text} target={target_text}")
        anchor_payload = getattr(head_map_result, "anchor_captured_payload", None)
        if isinstance(anchor_payload, dict):
            print(
                "Vivy head: right-thumbclick armed anchor captured "
                f"pan={anchor_payload.get('head_anchor_pan_degrees')} tilt={anchor_payload.get('head_anchor_tilt_degrees')} target={anchor_payload.get('head_target_joint_targets_deg')}"
            )

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

    @staticmethod
    def _extract_head_state(payload: dict) -> dict | None:
        head_state = payload.get("head_state")
        if isinstance(head_state, dict):
            return head_state
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
                    left_ik_joint_targets_deg = np.asarray(left_joint_targets_deg, dtype=float)[
                        self.left_controlled_to_ik_indices
                    ]
                    left_current_sim_pose = self.left_kinematics.forward_kinematics(
                        left_ik_joint_targets_deg
                    )
                    left_stage_pose = self.left_stage_io.read_stage_end_effector_pose(stage)
                    left_map_result = self.left_teleop_mapper.map_packet(
                        {"normalized": {"left_hand": left_hand}},
                        current_sim_pose=left_current_sim_pose,
                        current_stage_pose=left_stage_pose,
                        anchor_joint_targets_deg=np.asarray(left_joint_targets_deg, dtype=float),
                    )
                    if left_map_result is not None:
                        left_write_event = {
                            "timestamp": time.time(),
                            "type": "left_write",
                            "tracked": bool(getattr(left_map_result, "tracked", False)),
                            "follow_target_enabled": bool(getattr(left_map_result, "follow_target_enabled", False)),
                            "waiting_for_anchor": bool(getattr(left_map_result, "waiting_for_anchor", False)),
                        }
                        try:
                            left_ik_targets = self._solve_left_arm_targets(
                                current_joint_targets_deg=left_ik_joint_targets_deg,
                                target_pose_model=np.asarray(left_map_result.target_pose, dtype=float),
                            )
                            left_targets = self._merge_left_ik_joint_targets_deg(
                                np.asarray(left_joint_targets_deg, dtype=float),
                                left_ik_targets,
                            )
                            left_targets = self._apply_rotation_direct_overrides(
                                joint_names=list(self.left_stage_io.joint_names),
                                joint_targets_deg=left_targets,
                                hand_state=getattr(left_map_result, "hand_state", None),
                                quest_anchor_rotation=self.left_teleop_mapper.quest_anchor_rotation,
                            )
                            left_targets = self._apply_thumbstick_direct_overrides(
                                joint_names=list(self.left_stage_io.joint_names),
                                joint_targets_deg=left_targets,
                                hand_state=getattr(left_map_result, "hand_state", None),
                            )
                            left_targets = self._apply_button_pair_direct_overrides(
                                joint_names=list(self.left_stage_io.joint_names),
                                joint_targets_deg=left_targets,
                                hand_state=getattr(left_map_result, "hand_state", None),
                            )
                            self.left_stage_io.write_joint_targets_deg(stage, left_targets)
                            left_write_event["success"] = True
                            left_write_event["left_joint_names"] = list(self.left_stage_io.joint_names)
                            left_write_event["left_joint_targets_deg"] = [float(value) for value in left_targets.tolist()]
                        except Exception as exc:
                            left_write_event["success"] = False
                            left_write_event["error"] = str(exc)
                        try:
                            _append_sim_write_event(SIM_WRITE_EVENTS_PATH, left_write_event)
                        except Exception:
                            pass
                    self._log_left_mapping(left_map_result, left_hand)
        payload["left_hand_state"] = left_hand_payload
        payload["left_follow_target_enabled"] = None if left_map_result is None else left_map_result.follow_target_enabled
        payload["left_waiting_for_anchor"] = None if left_map_result is None else left_map_result.waiting_for_anchor
        payload["left_anchor_position"] = None if left_map_result is None else left_map_result.quest_anchor_position
        panel_payload["left_hand_state"] = left_hand_payload
        panel_payload["left_follow_target_enabled"] = None if left_map_result is None else left_map_result.follow_target_enabled
        panel_payload["left_waiting_for_anchor"] = None if left_map_result is None else left_map_result.waiting_for_anchor
        panel_payload["left_anchor_position"] = None if left_map_result is None else left_map_result.quest_anchor_position

        head_map_result = None
        head_state = None
        if stage is not None:
            head_state = self._extract_head_state(payload)
            if isinstance(head_state, dict):
                head_packet = {"head": dict(head_state)}
                normalized = payload.get("normalized")
                if isinstance(normalized, dict):
                    head_packet["normalized"] = dict(normalized)
                head_map_result = self.head_teleop_mapper.map_packet(head_packet)
                if head_map_result is not None:
                    head_write_event = {
                        "timestamp": time.time(),
                        "type": "head_write",
                        "tracked": bool(getattr(head_map_result, "tracked", False)),
                        "follow_target_enabled": bool(getattr(head_map_result, "follow_target_enabled", False)),
                        "waiting_for_anchor": bool(getattr(head_map_result, "waiting_for_anchor", False)),
                        "armed_by": getattr(head_map_result, "armed_by", None),
                    }
                    try:
                        head_targets = np.asarray(head_map_result.target_joint_targets_deg, dtype=float)
                        self.head_stage_io.write_joint_targets_deg(stage, head_targets)
                        head_write_event["success"] = True
                        head_write_event["head_joint_names"] = list(self.head_stage_io.joint_names)
                        head_write_event["head_joint_targets_deg"] = [float(value) for value in head_targets.tolist()]
                        print(
                            "Vivy head: stage=written "
                            f"head_pan={head_targets[0]:.2f} head_tilt={head_targets[1]:.2f}"
                        )
                    except Exception as exc:
                        head_write_event["success"] = False
                        head_write_event["error"] = str(exc)
                        print(f"Vivy head: stage write failed error={exc}")
                    try:
                        _append_sim_write_event(SIM_WRITE_EVENTS_PATH, head_write_event)
                    except Exception:
                        pass
                    self._log_head_mapping(head_map_result, head_state)
        payload["head_state"] = head_state
        payload["head_follow_target_enabled"] = None if head_map_result is None else head_map_result.follow_target_enabled
        payload["head_waiting_for_anchor"] = None if head_map_result is None else head_map_result.waiting_for_anchor
        payload["head_anchor_degrees"] = None if head_map_result is None else head_map_result.head_anchor_degrees.tolist() if head_map_result.head_anchor_degrees is not None else None
        payload["head_armed_by"] = None if head_map_result is None else head_map_result.armed_by
        panel_payload["head_state"] = head_state
        panel_payload["head_follow_target_enabled"] = None if head_map_result is None else head_map_result.follow_target_enabled
        panel_payload["head_waiting_for_anchor"] = None if head_map_result is None else head_map_result.waiting_for_anchor
        panel_payload["head_anchor_degrees"] = None if head_map_result is None else head_map_result.head_anchor_degrees.tolist() if head_map_result.head_anchor_degrees is not None else None
        panel_payload["head_armed_by"] = None if head_map_result is None else head_map_result.armed_by
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
