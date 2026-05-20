from __future__ import annotations

import importlib.util
import json
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation


SIM_CONFIG_PATH = Path("/home/viaan/huggingface/lerobot/src/lerobot/robots/vivy/vivy_global_config.json")
KINEMATICS_PATH = Path("/home/viaan/huggingface/lerobot/src/lerobot/robots/vivy/vivy_arm_kinematics.py")
TELEOP_STATE_PATH = Path("/home/viaan/huggingface/lerobot/src/lerobot/robots/vivy/target_stream/teleop_state.py")
TELEOP_DEBUG_VISUALS_PATH = Path("/home/viaan/vivy_hopejr_sim/ui/teleop_debug_visuals.py")
STAGE_IO_PATH = Path("/home/viaan/vivy_hopejr_sim/controllers/stage_io.py")
VIVY_SIDE_PANEL_PATH = Path("/home/viaan/vivy_hopejr_sim/ui/vivy/vivy_side_panel.py")
VIVY_FLOW_PANEL_PATH = Path("/home/viaan/vivy_hopejr_sim/ui/vivy/vivy_flow_panel.py")
VIVY_FLOW_DETAIL_PANEL_PATH = Path("/home/viaan/vivy_hopejr_sim/ui/vivy/vivy_flow_detail_panel.py")
CALIBRATION_PREVIEW_CONTROL_PATH = Path(
    "/home/viaan/vivy_hopejr_sim/controllers/vivy/calibration_preview_control.py"
)
CALIBRATION_LIMIT_FEEDBACK_PATH = Path(
    "/home/viaan/vivy_hopejr_sim/controllers/vivy/calibration_limit_feedback.py"
)
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


def _load_shared_signal_helpers():
    module = _load_module("vivy_target_viewer_teleop_state", TELEOP_STATE_PATH)
    return module.DEFAULT_TELEOP_STATE_PATH, module.TeleopStateUdpReceiver


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


_CALIBRATION_PREVIEW_CONTROL = _load_module(
    "vivy_calibration_preview_control",
    CALIBRATION_PREVIEW_CONTROL_PATH,
)
build_calibration_preview_control_state = _CALIBRATION_PREVIEW_CONTROL.build_calibration_preview_control_state
extract_calibration_limit_updates = _CALIBRATION_PREVIEW_CONTROL.extract_calibration_limit_updates
is_calibration_limit_request = _CALIBRATION_PREVIEW_CONTROL.is_calibration_limit_request
is_calibration_preview_payload = _CALIBRATION_PREVIEW_CONTROL.is_calibration_preview_payload
_CALIBRATION_LIMIT_FEEDBACK = _load_module(
    "vivy_calibration_limit_feedback",
    CALIBRATION_LIMIT_FEEDBACK_PATH,
)
CalibrationLimitFeedbackPublisher = _CALIBRATION_LIMIT_FEEDBACK.CalibrationLimitFeedbackPublisher
detect_limit_hits = _CALIBRATION_LIMIT_FEEDBACK.detect_limit_hits


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


class ControlStateUdpReceiver:
    def __init__(self, *, host: str, port: int):
        self.host = str(host)
        self.port = int(port)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((self.host, self.port))
        self._socket.setblocking(False)
        self._latest: dict | None = None

    def read_latest(self) -> dict | None:
        latest_payload = None
        while True:
            try:
                latest_payload, _addr = self._socket.recvfrom(1024 * 1024)
            except BlockingIOError:
                break
        if latest_payload is None:
            return self._latest
        try:
            payload = json.loads(latest_payload.decode("utf-8"))
        except Exception:
            return self._latest
        if isinstance(payload, dict) and payload.get("type") == "vivy_control_state":
            self._latest = payload
        return self._latest


class VivyTargetViewer:
    def __init__(self, *, signal_path: str | Path | None = None, interval_s: float = 0.05):
        default_signal_path, TeleopStateUdpReceiver = _load_shared_signal_helpers()
        VivyArmKinematics = _load_kinematics_class()
        HopeJrStageIo = _load_stage_io_class()
        TeleopDebugVisuals = _load_visuals_class()
        VivySidePanel = _load_side_panel_class()
        VivyFlowPanel = _load_flow_panel_class()
        VivyFlowDetailPanel = _load_flow_detail_panel_class()
        self.signal_path = Path(default_signal_path if signal_path is None else signal_path)
        self.interval_s = float(interval_s)
        self._last_tick_time = 0.0
        self._subscription = None
        self._last_signal_timestamp = None
        self._last_applied_feedback_timestamp: float | None = None
        self._last_applied_command_timestamp: float | None = None

        self.sim_config = _load_sim_config()
        controller_defaults = dict(self.sim_config.get("controller_defaults") or {})
        control_state_config = dict(controller_defaults.get("control_state") or {})
        target_state_config = dict(controller_defaults.get("target_state") or {})
        self.target_state_receiver = TeleopStateUdpReceiver(
            host=str(target_state_config.get("udp_host", "127.0.0.1")),
            port=int(target_state_config.get("sim_udp_port", 8771)),
        )
        self.vivy_event_receiver = TeleopStateUdpReceiver(
            host=str(target_state_config.get("udp_host", "127.0.0.1")),
            port=int(target_state_config.get("event_udp_port", 8777)),
            message_type="vivy_event",
        )
        self.control_state_receiver = ControlStateUdpReceiver(
            host=str(control_state_config.get("udp_host", "127.0.0.1")),
            port=int(control_state_config.get("udp_port", 8767)),
        )
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
        self.limits_by_joint = {
            name: {
                "min": float(config["min_deg"]),
                "max": float(config["max_deg"]),
            }
            for name, config in self.sim_config["joints"].items()
        }
        calibration_feedback_config = dict(controller_defaults.get("calibration_limit_feedback") or {})
        self.calibration_limit_feedback_publisher = CalibrationLimitFeedbackPublisher(
            host=str(calibration_feedback_config.get("udp_host", "127.0.0.1")),
            port=int(calibration_feedback_config.get("udp_port", 8778)),
        )
        ik_chains = dict(self.sim_config.get("ik_chains") or {})
        self.arm_sides = ("right", "left")
        self.arm_stage_ios = {}
        self.arm_joint_names = {}
        self.arm_end_effector_paths = {}
        for side in self.arm_sides:
            chain_config_for_side = dict(ik_chains.get(f"{side}_arm") or {})
            end_effector_config_for_side = dict(chain_config_for_side.get("end_effector") or {})
            default_end_effector_path = (
                self.end_effector_path
                if side == "right"
                else f"/World/JointTest/{side.capitalize()}Forearm/EndEffector"
            )
            end_effector_path_for_side = str(
                end_effector_config_for_side.get("frame_path", default_end_effector_path)
            )
            controlled_joint_names_for_side = list(chain_config_for_side.get("controlled_joint_names") or [])
            if not controlled_joint_names_for_side:
                controlled_joint_names_for_side = list(chain_config_for_side.get("joint_names") or self.joint_names)
            self.arm_joint_names[side] = list(controlled_joint_names_for_side)
            self.arm_end_effector_paths[side] = end_effector_path_for_side
        head_joint_names = ("head_pan", "head_tilt")
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
        self._stage_anchor_pose = None
        self._last_waiting_for_anchor = True
        for side in self.arm_sides:
            self.arm_stage_ios[side] = HopeJrStageIo(
                articulation_root_path=self.articulation_root_path,
                joint_root_path=self.joint_root_path,
                end_effector_path=self.arm_end_effector_paths[side],
                joint_names=self.arm_joint_names[side],
            )
        self.stage_io = self.arm_stage_ios["right"]
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
        self._seen_vivy_event_keys: set[tuple[int | None, str]] = set()
        self._calibration_initial_limits_published = False

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

    def _read_vivy_event_messages(self) -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        for event in self.vivy_event_receiver.read_all():
            message = str(event.get("event_message") or "").strip()
            if not message:
                continue
            try:
                timestamp_ns = int(event.get("timestamp_ns"))
            except (TypeError, ValueError):
                timestamp_ns = None
            key = (timestamp_ns, message)
            if key in self._seen_vivy_event_keys:
                continue
            self._seen_vivy_event_keys.add(key)
            events.append(event)
        return events

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

    def _write_arm_targets_from_payload(self, stage, payload: dict, side: str, *, teleop_changed: bool) -> dict | None:
        stage_io = self.arm_stage_ios.get(side)
        if stage is None or stage_io is None or not teleop_changed:
            return None
        follow_target_enabled = payload.get(f"{side}_follow_target_enabled")
        waiting_for_anchor = payload.get(f"{side}_waiting_for_anchor")
        joint_targets_deg = payload.get(f"{side}_current_joint_targets_deg")
        joint_names = payload.get(f"{side}_joint_names")
        hand_state = payload.get(f"{side}_hand_state")
        if side == "right":
            follow_target_enabled = payload.get("follow_target_enabled", follow_target_enabled)
            waiting_for_anchor = payload.get("waiting_for_anchor", waiting_for_anchor)
            joint_targets_deg = payload.get("current_joint_targets_deg", joint_targets_deg)
            joint_names = payload.get("joint_names", joint_names)
            hand_state = payload.get("hand_state", hand_state)
        if (
            not bool(follow_target_enabled)
            or bool(waiting_for_anchor)
            or not isinstance(joint_targets_deg, list)
            or not isinstance(joint_names, list)
        ):
            return None
        write_event = {
            "timestamp": time.time(),
            "type": f"{side}_write",
            "source": "target",
            "tracked": bool((hand_state or {}).get("is_tracked", False)),
            "follow_target_enabled": bool(follow_target_enabled),
            "waiting_for_anchor": bool(waiting_for_anchor),
        }
        try:
            if list(joint_names) != list(stage_io.joint_names):
                raise RuntimeError(
                    f"{side} joint name mismatch: target={joint_names} stage={list(stage_io.joint_names)}"
                )
            targets = np.asarray(joint_targets_deg, dtype=float)
            if targets.shape != (len(stage_io.joint_names),):
                raise RuntimeError(
                    f"{side} target shape mismatch: {targets.shape} expected {(len(stage_io.joint_names),)}"
                )
            if is_calibration_preview_payload(payload):
                limit_updates = extract_calibration_limit_updates(payload)
                if limit_updates:
                    write_event["calibration_limit_updates"] = {
                        joint_name: list(limits)
                        for joint_name, limits in stage_io.write_joint_limits_deg(stage, limit_updates).items()
                    }
                stage_limits_by_joint = stage_io.read_joint_limits_deg(stage)
                if stage_limits_by_joint:
                    limit_hits = detect_limit_hits(
                        joint_names=list(joint_names),
                        requested_deg=[float(value) for value in targets.tolist()],
                        limits_by_joint=stage_limits_by_joint,
                    )
                    publish_limits = not self._calibration_initial_limits_published
                    if publish_limits or limit_hits:
                        feedback_payload = self.calibration_limit_feedback_publisher.publish_status(
                            hits=limit_hits,
                            limits_by_joint=stage_limits_by_joint if publish_limits else None,
                            limit_source="isaac_stage",
                        )
                        if feedback_payload is not None:
                            write_event["calibration_limit_feedback"] = feedback_payload
                        if publish_limits:
                            self._calibration_initial_limits_published = True
                else:
                    write_event["calibration_limit_feedback"] = {
                        "success": False,
                        "reason": "isaac_stage_joint_limits_unavailable",
                    }
            stage_io.write_joint_targets_deg(stage, targets)
            write_event["success"] = True
            write_event[f"{side}_joint_names"] = list(stage_io.joint_names)
            write_event[f"{side}_joint_targets_deg"] = [float(value) for value in targets.tolist()]
            if side == "right":
                self._joint_write_ready = True
        except Exception as exc:
            write_event["success"] = False
            write_event["error"] = str(exc)
            if side == "right" and "Articulation unavailable" in str(exc):
                self._joint_write_ready = False
        try:
            _append_sim_write_event(SIM_WRITE_EVENTS_PATH, write_event)
        except Exception:
            pass
        return write_event

    def _apply_calibration_limit_updates(self, stage, payload: dict) -> dict | None:
        if stage is None or not is_calibration_preview_payload(payload):
            return None
        limit_updates = extract_calibration_limit_updates(payload)
        if not limit_updates:
            return None

        applied: dict[str, tuple[float, float]] = {}
        for stage_io in [*self.arm_stage_ios.values(), self.head_stage_io]:
            applied.update(stage_io.write_joint_limits_deg(stage, limit_updates))
        save_error = None
        if applied:
            try:
                self.stage_io.save_stage(stage)
            except Exception as exc:
                save_error = str(exc)
        stage_limits_by_joint: dict[str, tuple[float, float]] = {}
        for stage_io in [*self.arm_stage_ios.values(), self.head_stage_io]:
            stage_limits_by_joint.update(stage_io.read_joint_limits_deg(stage))
        update_status = {
            "success": bool(applied) and save_error is None,
            "updated_joints": list(applied),
            "saved_stage": save_error is None if applied else False,
        }
        if save_error is not None:
            update_status["error"] = save_error
        feedback_payload = self.calibration_limit_feedback_publisher.publish_status(
            hits=[],
            limits_by_joint=stage_limits_by_joint,
            limit_source="isaac_stage",
            limit_update_status=update_status,
        )
        event = {
            "timestamp": time.time(),
            "type": "calibration_limit_update",
            "success": bool(update_status["success"]),
            "requested": {joint_name: list(limits) for joint_name, limits in limit_updates.items()},
            "applied": {joint_name: list(limits) for joint_name, limits in applied.items()},
        }
        if save_error is not None:
            event["error"] = save_error
        if feedback_payload is not None:
            event["calibration_limit_feedback"] = feedback_payload
        try:
            _append_sim_write_event(SIM_WRITE_EVENTS_PATH, event)
        except Exception:
            pass
        return event

    def _publish_calibration_limit_snapshot(self, stage, payload: dict) -> dict | None:
        if stage is None or not is_calibration_limit_request(payload):
            return None
        stage_limits_by_joint: dict[str, tuple[float, float]] = {}
        for stage_io in [*self.arm_stage_ios.values(), self.head_stage_io]:
            stage_limits_by_joint.update(stage_io.read_joint_limits_deg(stage))
        feedback_payload = self.calibration_limit_feedback_publisher.publish_status(
            hits=[],
            limits_by_joint=stage_limits_by_joint,
            limit_source="isaac_stage",
        )
        self._calibration_initial_limits_published = True
        event = {
            "timestamp": time.time(),
            "type": "calibration_limit_snapshot",
            "success": bool(feedback_payload),
            "joint_count": len(stage_limits_by_joint),
        }
        if feedback_payload is not None:
            event["calibration_limit_feedback"] = feedback_payload
        try:
            _append_sim_write_event(SIM_WRITE_EVENTS_PATH, event)
        except Exception:
            pass
        return event

    def _collect_arm_payloads(self, payload: dict) -> dict[str, dict[str, object]]:
        arm_payloads: dict[str, dict[str, object]] = {}
        for side in self.arm_sides:
            hand_state = self._extract_hand_state(payload, side)
            follow_target_enabled = payload.get(f"{side}_follow_target_enabled")
            waiting_for_anchor = payload.get(f"{side}_waiting_for_anchor")
            anchor_position = payload.get(f"{side}_anchor_position")
            target_pose_model = payload.get(f"{side}_target_pose_model")
            if side == "right":
                hand_state = payload.get("hand_state", hand_state)
                follow_target_enabled = payload.get("follow_target_enabled", follow_target_enabled)
                waiting_for_anchor = payload.get("waiting_for_anchor", waiting_for_anchor)
                anchor_position = payload.get("quest_anchor_position", anchor_position)
                target_pose_model = payload.get("target_pose_model", target_pose_model)
            arm_payloads[side] = {
                "hand_state": hand_state,
                "follow_target_enabled": follow_target_enabled,
                "waiting_for_anchor": waiting_for_anchor,
                "anchor_position": anchor_position,
                "target_pose_model": target_pose_model,
                "sim_target_position": self._target_pose_model_to_stage_position(target_pose_model),
            }
        return arm_payloads

    def _sync_arm_panel_payload(self, payload: dict, panel_payload: dict, arm_payloads: dict[str, dict[str, object]]) -> None:
        for side, arm_payload in arm_payloads.items():
            if side == "right":
                continue
            payload[f"{side}_hand_state"] = arm_payload["hand_state"]
            payload[f"{side}_follow_target_enabled"] = arm_payload["follow_target_enabled"]
            payload[f"{side}_waiting_for_anchor"] = arm_payload["waiting_for_anchor"]
            payload[f"{side}_anchor_position"] = arm_payload["anchor_position"]
            panel_payload[f"{side}_hand_state"] = arm_payload["hand_state"]
            panel_payload[f"{side}_follow_target_enabled"] = arm_payload["follow_target_enabled"]
            panel_payload[f"{side}_waiting_for_anchor"] = arm_payload["waiting_for_anchor"]
            panel_payload[f"{side}_anchor_position"] = arm_payload["anchor_position"]

    def _target_pose_model_to_stage_position(self, target_pose_model: object) -> np.ndarray | None:
        if target_pose_model is None:
            return None
        try:
            target_pose_model_arr = np.asarray(target_pose_model, dtype=float)
            if target_pose_model_arr.shape != (4, 4):
                return None
            return (self._anchor_model_to_stage_transform @ target_pose_model_arr)[:3, 3] + self.world_offset
        except Exception:
            return None

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
            print(f"Vivy head: waiting_thumbclick current={current_text} anchor={anchor_text} target={target_text}")
            return
        state = "tracked" if bool(getattr(head_map_result, "tracked", False)) else "untracked"
        if bool(getattr(head_map_result, "follow_target_enabled", False)):
            state = f"{state}/armed"
        elif bool(getattr(head_map_result, "waiting_for_anchor", False)):
            state = f"{state}/waiting_thumbclick"
        if bool(getattr(head_map_result, "tracking_lost", False)):
            state = f"{state}/lost"
        print(f"Vivy head: {state} current={current_text} anchor={anchor_text} target={target_text}")
        anchor_payload = getattr(head_map_result, "anchor_captured_payload", None)
        if isinstance(anchor_payload, dict):
            armed_by = getattr(head_map_result, "armed_by", None) or "unknown"
            print(
                f"Vivy head: {armed_by}-thumbclick armed anchor captured "
                f"pan={anchor_payload.get('head_anchor_pan_degrees')} tilt={anchor_payload.get('head_anchor_tilt_degrees')} target={anchor_payload.get('head_target_joint_targets_deg')}"
            )

    @staticmethod
    def _extract_hand_state(payload: dict, side: str) -> dict | None:
        normalized = payload.get("normalized")
        if isinstance(normalized, dict):
            hand = normalized.get(f"{side}_hand")
            if isinstance(hand, dict):
                return hand
        parsed_message = payload.get("parsed_message")
        if isinstance(parsed_message, dict):
            side_title = side.capitalize()
            for key in (f"{side}_hand", side, f"{side}Controller", f"{side_title}Hand"):
                hand = parsed_message.get(key)
                if isinstance(hand, dict):
                    return hand
        return None

    @staticmethod
    def _extract_head_state(payload: dict) -> dict | None:
        head_state = payload.get("head_state")
        if isinstance(head_state, dict):
            return head_state
        return None

    def _apply_udp_control_state(self, payload: dict, panel_payload: dict, control_state: dict | None) -> None:
        if is_calibration_preview_payload(payload):
            preview_control_state = build_calibration_preview_control_state()
            payload["control_state"] = preview_control_state
            payload["control_state_source"] = "calibration_preview"
            panel_payload["control_state"] = dict(preview_control_state)
            panel_payload["control_state_source"] = "calibration_preview"
            return
        if isinstance(control_state, dict):
            payload["control_state"] = dict(control_state)
            payload["control_state_source"] = "udp"
            panel_payload["control_state"] = dict(control_state)
        else:
            payload["control_state"] = {
                "type": "vivy_control_state",
                "source": "udp",
                "status": "waiting_for_udp",
                "right": {"enabled": False},
                "left": {"enabled": False},
                "head": {"enabled": False, "armed_by": None},
            }
            payload["control_state_source"] = "udp_waiting"
            panel_payload["control_state"] = dict(payload["control_state"])
            control_state = payload["control_state"]
        for side in self.arm_sides:
            side_state = control_state.get(side)
            if not isinstance(side_state, dict):
                side_state = {}
            follow_enabled = bool(side_state.get("enabled", False))
            payload[f"{side}_follow_target_enabled"] = follow_enabled
            panel_payload[f"{side}_follow_target_enabled"] = follow_enabled
            if side == "right":
                payload["follow_target_enabled"] = follow_enabled
                panel_payload["follow_target_enabled"] = follow_enabled
        head_state = control_state.get("head")
        if not isinstance(head_state, dict):
            head_state = {}
        head_enabled = bool(head_state.get("enabled", False))
        payload["head_follow_target_enabled"] = head_enabled
        payload["head_waiting_for_anchor"] = not head_enabled
        payload["head_armed_by"] = head_state.get("armed_by")
        panel_payload["head_follow_target_enabled"] = head_enabled
        panel_payload["head_waiting_for_anchor"] = not head_enabled
        panel_payload["head_armed_by"] = head_state.get("armed_by")
        head_control = payload.get("head_control")
        if isinstance(head_control, dict):
            head_control = dict(head_control)
            head_control["follow_target_enabled"] = head_enabled
            head_control["waiting_for_anchor"] = not head_enabled
            head_control["armed_by"] = head_state.get("armed_by")
            head_control["event_seq"] = head_state.get("event_seq")
            payload["head_control"] = head_control

    def _on_update(self, _event: object) -> None:
        now = time.monotonic()
        if now - self._last_tick_time < self.interval_s:
            return
        self._last_tick_time = now

        stage = self._read_stage()
        stage_pose = self._maybe_refresh_transform(stage)
        if stage is None or stage_pose is None:
            return

        payload = self.target_state_receiver.read_latest()
        if not isinstance(payload, dict):
            vivy_events = self._read_vivy_event_messages()
            if vivy_events:
                try:
                    self.side_panel.update({"event_messages": vivy_events})
                except Exception as exc:
                    print(f"Vivy side panel event-only update failed: {exc}")
            return
        flow_control = _read_flow_control()
        real_feedback = self._read_real_feedback()
        real_joint_positions_deg = self._read_real_feedback_joint_positions_deg(real_feedback)
        panel_payload = self._inject_real_feedback_rows(payload, real_feedback)
        vivy_events = self._read_vivy_event_messages()
        if vivy_events:
            existing_events = panel_payload.get("event_messages")
            if isinstance(existing_events, list):
                panel_payload["event_messages"] = [*vivy_events, *existing_events]
            else:
                panel_payload["event_messages"] = vivy_events
        control_state = self.control_state_receiver.read_latest()
        self._apply_udp_control_state(payload, panel_payload, control_state)
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
        if not stage_write_blocked:
            if real_joint_positions_deg is not None:
                if feedback_changed:
                    self._write_sim_joint_positions(stage, np.asarray(real_joint_positions_deg, dtype=float), update_state=False)
                    self._last_applied_feedback_timestamp = feedback_timestamp

        if teleop_changed:
            self._last_signal_timestamp = signal_timestamp
        else:
            return

        self._publish_calibration_limit_snapshot(stage, payload)
        self._apply_calibration_limit_updates(stage, payload)

        for side in self.arm_sides:
            if side == "right" and (stage_write_blocked or real_joint_positions_deg is not None):
                continue
            write_event = self._write_arm_targets_from_payload(stage, payload, side, teleop_changed=teleop_changed)
            if side == "right" and write_event and write_event.get("success"):
                self._last_applied_command_timestamp = signal_timestamp

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
        arm_payloads = self._collect_arm_payloads(payload)
        self._sync_arm_panel_payload(payload, panel_payload, arm_payloads)

        head_map_result = None
        head_state = None
        if stage is not None:
            head_state = self._extract_head_state(payload)
            head_control = payload.get("head_control")
            head_targets_payload = payload.get("head_joint_targets_deg")
            if isinstance(head_control, dict) and isinstance(head_targets_payload, list):
                head_write_event = {
                    "timestamp": time.time(),
                    "type": "head_write",
                    "tracked": bool(head_control.get("tracked", False)),
                    "follow_target_enabled": bool(head_control.get("follow_target_enabled", False)),
                    "waiting_for_anchor": bool(head_control.get("waiting_for_anchor", False)),
                    "armed_by": head_control.get("armed_by"),
                }
                try:
                    head_targets = np.asarray(head_targets_payload, dtype=float)
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
                self._log_head_mapping(SimpleNamespace(**head_control), head_state or {})
            elif isinstance(head_state, dict):
                print("Vivy head: waiting for shared head_control from target")
        payload["head_state"] = head_state
        payload["head_follow_target_enabled"] = payload.get("head_follow_target_enabled")
        payload["head_waiting_for_anchor"] = payload.get("head_waiting_for_anchor")
        payload["head_anchor_degrees"] = payload.get("head_anchor_degrees")
        payload["head_armed_by"] = (payload.get("head_control") or {}).get("armed_by") if isinstance(payload.get("head_control"), dict) else None
        panel_payload["head_state"] = head_state
        panel_payload["head_follow_target_enabled"] = payload.get("head_follow_target_enabled")
        panel_payload["head_waiting_for_anchor"] = payload.get("head_waiting_for_anchor")
        panel_payload["head_anchor_degrees"] = payload.get("head_anchor_degrees")
        panel_payload["head_armed_by"] = payload.get("head_armed_by")
        try:
            self.side_panel.update(panel_payload)
        except Exception as exc:
            print(f"Vivy side panel update failed: {exc}")
        show_pitch_frames = bool(flow_control.get("show_pitch_frames", False))
        pitch_visual = self._build_pitch_visual(stage) if show_pitch_frames else None
        arm_visuals = {}
        for side, arm_payload in arm_payloads.items():
            sim_target_position_side = arm_payload.get("sim_target_position")
            if sim_target_position_side is None:
                continue
            arm_visuals[side] = {
                "quest_mapped_position": sim_target_position_side,
                "sim_target_position": sim_target_position_side,
                "waiting_for_anchor": arm_payload.get("waiting_for_anchor"),
            }
        self.visuals.enabled = bool(flow_control.get("sim_view_enabled", True))
        self.visuals.update(
            stage,
            quest_anchor_position=np.asarray(payload.get("quest_anchor_position") or [0.0, 0.0, 0.0], dtype=float),
            quest_current_position=np.asarray((payload.get("hand_state") or {}).get("position") or [0.0, 0.0, 0.0], dtype=float),
            quest_mapped_position=quest_mapped_position,
            sim_target_position=sim_target_position,
            arm_visuals=arm_visuals,
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
