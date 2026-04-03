#!/usr/bin/env python3

import argparse
import importlib.util
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ui.hope_jr_teleop_status_ui import HopeJrTeleopStatusUi
from ui.teleop_debug_visuals import TeleopDebugVisuals
from controllers.quest_teleop_mapper import QuestTeleopMapper
from controllers.joint_control_policy import (
    DEFAULT_POSITION_ONLY_JOINT_CONTROL_PROFILE,
    JOINT_CONTROL_HOLD_LAST,
    JOINT_CONTROL_HOLD_START,
    build_joint_control_modes,
)
from controllers.joint_target_conditioner import JointTargetConditioner
from controllers.teleop_packet_source import TeleopPacketSource
from controllers.guards.heuristic_safety_guard import HeuristicSafetyGuard
from controllers.guards.joint_limit_safety_guard import JointLimitSafetyGuard
from controllers.stage_io import HopeJrStageIo

import numpy as np
from scipy.spatial.transform import Rotation


DEFAULT_LEROBOT_REPO = Path("/home/viaan/huggingface/lerobot")
DEFAULT_SIM_CONFIG_PATH = DEFAULT_LEROBOT_REPO / "src/lerobot/robots/vivy/vivy_global_config.json"
DEFAULT_KINEMATICS_MODULE_PATH = DEFAULT_LEROBOT_REPO / "src/lerobot/robots/vivy/vivy_arm_kinematics.py"
DEFAULT_ARTICULATION_ROOT_PATH = "/World/JointTest"
DEFAULT_JOINT_ROOT_PATH = "/World/JointTest/Joints"
DEFAULT_UDP_LISTEN_HOST = "127.0.0.1"
DEFAULT_UDP_LISTEN_PORT = 8766
DEFAULT_DEBUG_PATH = Path("/tmp/hope_jr_sim_ik_debug.json")
DEFAULT_TELEOP_DEBUG_ROOT = "/World/JointTest/TeleopDebug"
DEFAULT_END_EFFECTOR_PATH = "/World/JointTest/PalmBody/EndEffector"
DEFAULT_EVENT_LOG_PATH = Path("/tmp/hope_jr_sim_ik_events.ndjson")
DEFAULT_PACKET_STALE_TIMEOUT_S = 0.75
DEFAULT_SIM_PLAY_STATE_PATH = Path("/tmp/hope_jr_sim_play_state.json")
DEFAULT_STOP_TARGETS_DEG = {}
DEFAULT_STAGE_DLS_LAMBDA = 0.01
DEFAULT_STAGE_DLS_MAX_STEP_DEG = 8.0
DEFAULT_STAGE_TASK_DELTA_CLAMP_M = 0.03
DEFAULT_STAGE_ERROR_SCORE_WINDOW = 120
DEFAULT_JOINT_TARGET_MAX_DELTA_DEG_PER_TICK = 2.0
DEFAULT_LIMIT_PUSH_FREEZE_CONSECUTIVE_FRAMES = 3
DEFAULT_STAGE_POSITION_ONLY_WEIGHT_OVERRIDES = {
    "right_forearm_twist": 0.0,
    "right_wrist": 0.0,
    "right_palm": 0.0,
}

_ACTIVE_LOOP = None


def _load_neutral_pose_map_from_config(config: dict[str, Any]) -> dict[str, float]:
    joints = config.get("joints")
    if not isinstance(joints, dict):
        return {}
    stop_targets: dict[str, float] = {}
    for key, value in joints.items():
        if not isinstance(value, dict) or "neutral_deg" not in value:
            continue
        try:
            stop_targets[str(key)] = float(value["neutral_deg"])
        except (TypeError, ValueError):
            continue
    return stop_targets


def _load_sim_config(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        raise RuntimeError(f"Failed to load Hope global config from {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Hope global config must be a JSON object: {path}")
    return data


def _build_stage_joint_weights_from_config(config: dict[str, Any], joint_names: list[str]) -> list[float]:
    joints = config.get("joints")
    if not isinstance(joints, dict):
        raise RuntimeError("Hope global config missing object field: joints")
    weights: list[float] = []
    missing: list[str] = []
    for name in joint_names:
        joint_cfg = joints.get(name)
        if not isinstance(joint_cfg, dict) or "weight" not in joint_cfg:
            missing.append(name)
            continue
        try:
            weights.append(float(joint_cfg["weight"]))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid weight for joint {name!r} in Hope global config") from exc
    if missing:
        missing_text = ", ".join(missing)
        raise RuntimeError(f"Missing joint weights in Hope global config for: {missing_text}")
    return weights


def _build_joint_limits_from_config(config: dict[str, Any], joint_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    joints = config.get("joints")
    if not isinstance(joints, dict):
        raise RuntimeError("Hope global config missing object field: joints")
    lower_limits: list[float] = []
    upper_limits: list[float] = []
    missing: list[str] = []
    for name in joint_names:
        joint_cfg = joints.get(name)
        if not isinstance(joint_cfg, dict) or "min_deg" not in joint_cfg or "max_deg" not in joint_cfg:
            missing.append(name)
            continue
        try:
            lower_limits.append(float(joint_cfg["min_deg"]))
            upper_limits.append(float(joint_cfg["max_deg"]))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid min/max for joint {name!r} in Hope global config") from exc
    if missing:
        missing_text = ", ".join(missing)
        raise RuntimeError(f"Missing joint limits in Hope global config for: {missing_text}")
    return np.asarray(lower_limits, dtype=float), np.asarray(upper_limits, dtype=float)


def _get_config_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        return {}
    return dict(value)


def _load_repo_sim_config(lerobot_repo: str | Path) -> dict[str, Any]:
    repo_path = Path(lerobot_repo)
    config_path = DEFAULT_SIM_CONFIG_PATH if repo_path == DEFAULT_LEROBOT_REPO else repo_path / "src/lerobot/robots/vivy/vivy_global_config.json"
    return _load_sim_config(config_path)


def _build_run_log_path(base_path: Path, run_id: str) -> Path:
    return base_path.with_name(f"{base_path.stem}_{run_id}{base_path.suffix}")


class HopeJrSimIkController:
    def __init__(
        self,
        *,
        lerobot_repo: Path,
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
    ):
        self.lerobot_repo = lerobot_repo
        self.joint_root_path = joint_root_path.rstrip("/")
        self.articulation_root_path = self.joint_root_path.rsplit("/", 1)[0]
        self.position_only = position_only
        self.debug_path = debug_path
        self.latest_debug_path = debug_path

        self.teleop_debug_root = teleop_debug_root.rstrip("/")
        self.show_teleop_debug = show_teleop_debug
        self.anchor_delay_s = anchor_delay_s
        self.event_log_path = event_log_path
        self.latest_event_log_path = event_log_path
        self.packet_stale_timeout_s = float(packet_stale_timeout_s)
        self.end_effector_path = end_effector_path
        self.write_joint_state_directly = bool(write_joint_state_directly)
        self.packet_source = TeleopPacketSource(
            use_udp=use_udp,
            udp_listen_host=udp_listen_host,
            udp_listen_port=udp_listen_port,
        )
        self.kinematics_module = self._load_kinematics_module(
            DEFAULT_KINEMATICS_MODULE_PATH
            if lerobot_repo == DEFAULT_LEROBOT_REPO
            else lerobot_repo / "src/lerobot/robots/vivy/vivy_arm_kinematics.py"
        )
        self.sim_config = _load_sim_config(
            DEFAULT_SIM_CONFIG_PATH
            if lerobot_repo == DEFAULT_LEROBOT_REPO
            else lerobot_repo / "src/lerobot/robots/vivy/vivy_global_config.json"
        )
        self.safety_guard_config = _get_config_section(self.sim_config, "safety_guards")
        self.heuristic_safety_guard = HeuristicSafetyGuard(self.safety_guard_config.get("heuristic"))
        self.joint_limit_safety_guard = JointLimitSafetyGuard(self.safety_guard_config.get("joint_limit"))
        self.model = self.kinematics_module.HopeJrArmKinematics.from_json(
            DEFAULT_SIM_CONFIG_PATH
            if lerobot_repo == DEFAULT_LEROBOT_REPO
            else lerobot_repo / "src/lerobot/robots/vivy/vivy_global_config.json"
        )
        self.controller_defaults = _get_config_section(self.sim_config, "controller_defaults")
        self.script_editor_test_defaults = _get_config_section(self.sim_config, "script_editor_test_defaults")
        self.stop_targets_deg = self._load_stop_targets_deg(self.sim_config)
        self.stage_joint_weights = np.asarray(
            _build_stage_joint_weights_from_config(self.sim_config, list(self.model.joint_names)),
            dtype=float,
        )
        self.lower_joint_limits_deg, self.upper_joint_limits_deg = _build_joint_limits_from_config(
            self.sim_config,
            list(self.model.joint_names),
        )
        self.position_only_joint_control_profile = str(self.controller_defaults.get("position_only_joint_control_profile", DEFAULT_POSITION_ONLY_JOINT_CONTROL_PROFILE))
        self.position_only_joint_control_modes = build_joint_control_modes(
            self.model.joint_names, self.position_only_joint_control_profile
        )
        self.last_joint_targets_deg = np.zeros(len(self.model.joint_names), dtype=float)
        self.last_commanded_stage_joint_targets_deg = None
        self.neutral_model_joint_targets_deg = np.array([self.stop_targets_deg.get(name, 0.0) for name in self.model.joint_names], dtype=float)
        self.start_stage_joint_positions_deg = None
        self.last_packet_timestamp = None
        self.position_only_weight_overrides = {str(k): float(v) for k, v in self.controller_defaults.get("position_only_weight_overrides", DEFAULT_STAGE_POSITION_ONLY_WEIGHT_OVERRIDES).items()}
        self.stage_dls_lambda = float(self.controller_defaults.get("stage_dls_lambda", DEFAULT_STAGE_DLS_LAMBDA))
        self.stage_dls_max_step_deg = float(self.controller_defaults.get("stage_dls_max_step_deg", DEFAULT_STAGE_DLS_MAX_STEP_DEG))
        self.stage_task_delta_clamp_m = float(self.controller_defaults.get("stage_task_delta_clamp_m", DEFAULT_STAGE_TASK_DELTA_CLAMP_M))
        self.stage_error_score_window = int(self.controller_defaults.get("stage_error_score_window", DEFAULT_STAGE_ERROR_SCORE_WINDOW))
        self.joint_target_max_delta_deg_per_tick = float(
            self.controller_defaults.get("joint_target_max_delta_deg_per_tick", DEFAULT_JOINT_TARGET_MAX_DELTA_DEG_PER_TICK)
        )
        self.limit_push_freeze_consecutive_frames = int(
            self.controller_defaults.get(
                "limit_push_freeze_consecutive_frames",
                DEFAULT_LIMIT_PUSH_FREEZE_CONSECUTIVE_FRAMES,
            )
        )
        self.joint_target_conditioner = JointTargetConditioner(
            max_delta_deg_per_tick=self.joint_target_max_delta_deg_per_tick
        )
        self.stage_error_norm_window = deque(maxlen=self.stage_error_score_window)
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
        self.teleop_debug_visuals = TeleopDebugVisuals(teleop_debug_root=self.teleop_debug_root, enabled=self.show_teleop_debug)
        self._a_pressed_last = False
        self.last_hand_state = {}
        self.last_packet_received_at = None
        self.limit_push_freeze_active = False
        self.limit_push_freeze_streak = 0
        self.limit_push_freeze_payload: dict[str, Any] | None = None
        self.stage_io = HopeJrStageIo(
            articulation_root_path=self.articulation_root_path,
            joint_root_path=self.joint_root_path,
            end_effector_path=self.end_effector_path,
            joint_names=self.model.joint_names,
        )

    def _load_kinematics_module(self, module_path: Path):
        spec = importlib.util.spec_from_file_location("hope_jr_arm_kinematics", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load Hope Jr kinematics module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def _load_latest_packet(self) -> dict[str, Any] | None:
        return self.packet_source.read_latest_packet()

    def _load_stop_targets_deg(self, sim_config: dict[str, Any]) -> dict[str, float]:
        stop_targets = dict(DEFAULT_STOP_TARGETS_DEG)
        stop_targets.update(_load_neutral_pose_map_from_config(sim_config))
        return stop_targets

    @staticmethod
    def _clip_vector_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= max_norm or norm <= 1e-12:
            return vector
        return vector * (max_norm / norm)

    def _compute_stage_error_score(self, stage_end_effector_error: list[float] | None) -> dict[str, Any] | None:
        if stage_end_effector_error is not None:
            error_norm = float(np.linalg.norm(np.asarray(stage_end_effector_error, dtype=float)))
            self.stage_error_norm_window.append(error_norm)
        if not self.stage_error_norm_window:
            return None
        values = np.asarray(self.stage_error_norm_window, dtype=float)
        return {
            "window_size": int(values.size),
            "mean_error_norm_m": float(values.mean()),
            "max_error_norm_m": float(values.max()),
            "latest_error_norm_m": float(values[-1]),
        }


    def _packet_to_target_pose(
        self,
        packet: dict[str, Any],
        current_joint_targets_deg: np.ndarray,
    ) -> Any | None:
        anchor_joint_targets_deg = current_joint_targets_deg.copy()
        current_sim_pose = self.model.forward_kinematics(anchor_joint_targets_deg)
        stage = self.stage_io.get_stage()
        current_stage_pose = self.stage_io.read_stage_end_effector_pose(stage)
        map_result = self.teleop_mapper.map_packet(
            packet,
            current_sim_pose=current_sim_pose,
            current_stage_pose=current_stage_pose,
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
                actual_end_effector_position=self.stage_io.read_stage_end_effector_position(stage),
                actual_end_effector_pose=self.stage_io.read_stage_end_effector_pose(stage),
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
            actual_end_effector_position=self.stage_io.read_stage_end_effector_position(stage),
            actual_end_effector_pose=self.stage_io.read_stage_end_effector_pose(stage),
        )
        return map_result


    def _append_event(self, payload: dict[str, Any], *, dedupe_key: tuple[Any, ...] | None = None) -> None:
        if dedupe_key is not None and dedupe_key == self._last_event_key:
            return
        self._last_event_key = dedupe_key
        event = {"logged_at": time.time(), **payload}
        serialized = json.dumps(event) + "\n"
        for path in [self.event_log_path, self.latest_event_log_path]:
            if path is None:
                continue
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(serialized)
            except Exception:
                continue

    def _write_debug(self, payload: dict[str, Any]) -> None:
        self.last_debug_payload = payload
        serialized = json.dumps(payload, indent=2) + "\n"
        for path in [self.debug_path, self.latest_debug_path]:
            if path is None:
                continue
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = path.with_suffix(".tmp")
                tmp_path.write_text(serialized)
                tmp_path.replace(path)
            except Exception:
                continue

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
        stage = self.stage_io.get_stage()
        self.teleop_debug_visuals.update(
            stage,
            quest_anchor_position=quest_anchor_position,
            quest_current_position=quest_current_position,
            quest_mapped_position=quest_mapped_position,
            sim_target_position=sim_target_position,
            reference_position=None if self.teleop_mapper.stage_anchor_pose is None else self.teleop_mapper.stage_anchor_pose[:3, 3],
            actual_end_effector_position=actual_end_effector_position,
            actual_end_effector_pose=actual_end_effector_pose,
            waiting_for_anchor=waiting_for_anchor,
        )


    def reset_target_positions(self, target_value_deg: float = 0.0, reset_joint_state: bool = True) -> None:
        self.teleop_mapper.reset()
        self.limit_push_freeze_active = False
        self.limit_push_freeze_streak = 0
        self.limit_push_freeze_payload = None
        stage = self.stage_io.get_stage()
        if stage is None:
            return
        target_values = np.array(
            [self.stop_targets_deg.get(joint_name, float(target_value_deg)) for joint_name in self.model.joint_names],
            dtype=float,
        )
        self.stage_io.write_joint_targets_deg(stage, target_values)
        if reset_joint_state:
            self.stage_io.write_joint_state_deg(stage, target_values)
        self.last_joint_targets_deg = self.stage_io.stage_to_model_joint_positions_deg(target_values)
        self.last_commanded_stage_joint_targets_deg = np.asarray(target_values, dtype=float).copy()
        self.start_stage_joint_positions_deg = np.asarray(target_values, dtype=float).copy()

    def _check_limit_push_freeze(
        self,
        *,
        current_stage_joint_positions_deg: np.ndarray | None,
        proposed_joint_targets_deg: np.ndarray,
    ) -> dict[str, Any] | None:
        if current_stage_joint_positions_deg is None:
            self.limit_push_freeze_streak = 0
            return None
        current = np.asarray(current_stage_joint_positions_deg, dtype=float)
        proposed = np.asarray(proposed_joint_targets_deg, dtype=float)
        lower = self.lower_joint_limits_deg
        upper = self.upper_joint_limits_deg
        saturation_tol_deg = 0.5
        push_tol_deg = 0.25
        for index, joint_name in enumerate(self.model.joint_names):
            current_deg = float(current[index])
            target_deg = float(proposed[index])
            lower_limit_deg = float(lower[index])
            upper_limit_deg = float(upper[index])
            pinned_upper = current_deg >= (upper_limit_deg - saturation_tol_deg)
            pinned_lower = current_deg <= (lower_limit_deg + saturation_tol_deg)
            pushing_upper = target_deg > max(upper_limit_deg + push_tol_deg, current_deg + push_tol_deg)
            pushing_lower = target_deg < min(lower_limit_deg - push_tol_deg, current_deg - push_tol_deg)
            if pinned_upper and pushing_upper:
                self.limit_push_freeze_streak += 1
                return {
                    "active": True,
                    "source": "output_freeze",
                    "source_label": "Output freeze",
                    "severity": "critical",
                    "reasons": [
                        "Joint is pinned at its upper limit while IK keeps pushing farther",
                        f"Freeze triggered after {self.limit_push_freeze_streak} consecutive limit-push frames",
                    ],
                    "recommendations": ["Press A to reset and re-anchor"],
                    "joint_name": joint_name,
                    "joint_index": index,
                    "current_joint_deg": current_deg,
                    "target_joint_deg": target_deg,
                    "lower_limit_deg": lower_limit_deg,
                    "upper_limit_deg": upper_limit_deg,
                    "freeze_trigger": "upper_limit_push",
                    "consecutive_frames": self.limit_push_freeze_streak,
                }
            if pinned_lower and pushing_lower:
                self.limit_push_freeze_streak += 1
                return {
                    "active": True,
                    "source": "output_freeze",
                    "source_label": "Output freeze",
                    "severity": "critical",
                    "reasons": [
                        "Joint is pinned at its lower limit while IK keeps pushing farther",
                        f"Freeze triggered after {self.limit_push_freeze_streak} consecutive limit-push frames",
                    ],
                    "recommendations": ["Press A to reset and re-anchor"],
                    "joint_name": joint_name,
                    "joint_index": index,
                    "current_joint_deg": current_deg,
                    "target_joint_deg": target_deg,
                    "lower_limit_deg": lower_limit_deg,
                    "upper_limit_deg": upper_limit_deg,
                    "freeze_trigger": "lower_limit_push",
                    "consecutive_frames": self.limit_push_freeze_streak,
                }
        self.limit_push_freeze_streak = 0
        return None

    def _solve_stage_differential_ik(
        self,
        *,
        stage,
        current_stage_joint_positions_deg: np.ndarray,
        target_pose: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        jacobian = self.stage_io.compute_end_effector_jacobian(stage)
        current_stage_pose = self.stage_io.read_stage_end_effector_pose(stage)
        if jacobian is None or current_stage_pose is None:
            return None

        target_stage_pose = self.teleop_mapper.model_to_stage_transform @ target_pose
        raw_position_error = target_stage_pose[:3, 3] - current_stage_pose[:3, 3]
        position_error = self._clip_vector_norm(raw_position_error, self.stage_task_delta_clamp_m)
        if self.position_only:
            task_error = position_error
            task_jacobian = jacobian[:3, :]
        else:
            rotation_error = Rotation.from_matrix(target_stage_pose[:3, :3] @ current_stage_pose[:3, :3].T).as_rotvec()
            task_error = np.concatenate([position_error, rotation_error])
            task_jacobian = jacobian

        solve_joint_weights = self.stage_joint_weights.copy()
        if self.position_only:
            solve_joint_weights = solve_joint_weights.copy()
            for index, name in enumerate(self.model.joint_names):
                override = self.position_only_weight_overrides.get(name)
                if override is not None:
                    solve_joint_weights[index] = min(solve_joint_weights[index], float(override))
        task_jacobian = task_jacobian * solve_joint_weights[None, :]

        damping = self.stage_dls_lambda
        try:
            delta_rad = task_jacobian.T @ np.linalg.solve(
                task_jacobian @ task_jacobian.T + (damping**2) * np.eye(task_jacobian.shape[0]),
                task_error,
            )
        except np.linalg.LinAlgError:
            return None

        delta_deg = np.rad2deg(delta_rad) * solve_joint_weights
        unclipped_delta_deg = delta_deg.copy()
        delta_deg = np.clip(delta_deg, -self.stage_dls_max_step_deg, self.stage_dls_max_step_deg)
        solved_stage_joint_targets_deg = np.asarray(current_stage_joint_positions_deg, dtype=float) + delta_deg
        return solved_stage_joint_targets_deg, target_stage_pose, delta_deg, unclipped_delta_deg, raw_position_error, position_error, solve_joint_weights

    def _apply_joint_control_policy(
        self,
        *,
        current_stage_joint_positions_deg: np.ndarray | None,
        solved_stage_joint_targets_deg: np.ndarray,
    ) -> tuple[np.ndarray, list[str]]:
        final_targets = np.asarray(solved_stage_joint_targets_deg, dtype=float).copy()
        if not self.position_only:
            return final_targets, ["solve"] * len(final_targets)

        joint_modes = list(self.position_only_joint_control_modes)
        start_targets = self.start_stage_joint_positions_deg
        last_targets = self.last_commanded_stage_joint_targets_deg
        current_targets = np.asarray(current_stage_joint_positions_deg, dtype=float) if current_stage_joint_positions_deg is not None else None
        for index, mode in enumerate(joint_modes):
            if mode == JOINT_CONTROL_HOLD_START and start_targets is not None:
                final_targets[index] = float(start_targets[index])
            elif mode == JOINT_CONTROL_HOLD_LAST:
                if last_targets is not None:
                    final_targets[index] = float(last_targets[index])
                elif start_targets is not None:
                    final_targets[index] = float(start_targets[index])
                elif current_targets is not None:
                    final_targets[index] = float(current_targets[index])
        return final_targets, joint_modes

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

        stage = self.stage_io.get_stage() if apply_to_stage else None
        current_stage_joint_positions_deg = None
        if stage is not None:
            current_stage_joint_positions_deg = self.stage_io.read_stage_joint_positions_deg(stage)
            if current_stage_joint_positions_deg is None:
                current_stage_joint_positions_deg = self.stage_io.read_current_joint_targets_deg(stage)
            current_joint_targets_deg = self.stage_io.stage_to_model_joint_positions_deg(current_stage_joint_positions_deg)
        else:
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

        target_stage_pose = self.teleop_mapper.model_to_stage_transform @ target_pose
        solved_joint_targets_deg = None
        stage_dls_delta_deg = None
        stage_dls_unclipped_delta_deg = None
        stage_dls_raw_position_error = None
        stage_dls_clamped_position_error = None
        stage_dls_joint_weights = None
        stage_dls_debug = None
        if stage is not None and current_stage_joint_positions_deg is not None:
            if self.start_stage_joint_positions_deg is None:
                self.start_stage_joint_positions_deg = np.asarray(current_stage_joint_positions_deg, dtype=float).copy()
            stage_solve = self._solve_stage_differential_ik(
                stage=stage,
                current_stage_joint_positions_deg=current_stage_joint_positions_deg,
                target_pose=target_pose,
            )
            stage_dls_debug = dict(self.stage_io.last_stage_dls_debug)
            if stage_solve is not None:
                solved_joint_targets_deg, target_stage_pose, stage_dls_delta_deg, stage_dls_unclipped_delta_deg, stage_dls_raw_position_error, stage_dls_clamped_position_error, stage_dls_joint_weights = stage_solve
        elif stage is not None:
            stage_dls_debug = {"reason": "current_stage_joint_positions_unavailable"}

        if stage is not None and solved_joint_targets_deg is None:
            stage_dls_unavailable_event = {
                "status": "ignored",
                "reason": "stage_dls_unavailable",
                "packet_timestamp": packet_timestamp,
                "stage_dls_debug": stage_dls_debug,
            }
            self._write_debug(stage_dls_unavailable_event)
            return None

        if solved_joint_targets_deg is None:
            solved_joint_targets_deg = np.asarray(current_stage_joint_positions_deg, dtype=float).copy() if current_stage_joint_positions_deg is not None else self.stage_io.model_to_stage_joint_positions_deg(current_joint_targets_deg)

        solved_joint_targets_deg, joint_control_modes = self._apply_joint_control_policy(
            current_stage_joint_positions_deg=current_stage_joint_positions_deg,
            solved_stage_joint_targets_deg=solved_joint_targets_deg,
        )
        limit_push_freeze = self._check_limit_push_freeze(
            current_stage_joint_positions_deg=current_stage_joint_positions_deg,
            proposed_joint_targets_deg=solved_joint_targets_deg,
        )
        if self.limit_push_freeze_active:
            limit_push_freeze = self.limit_push_freeze_payload
        elif limit_push_freeze is not None and int(limit_push_freeze.get("consecutive_frames", 0)) >= self.limit_push_freeze_consecutive_frames:
            self.limit_push_freeze_active = True
            self.limit_push_freeze_payload = dict(limit_push_freeze)
            limit_push_freeze = self.limit_push_freeze_payload
        if self.limit_push_freeze_active:
            if self.last_commanded_stage_joint_targets_deg is not None:
                solved_joint_targets_deg = np.asarray(self.last_commanded_stage_joint_targets_deg, dtype=float).copy()
            elif current_stage_joint_positions_deg is not None:
                solved_joint_targets_deg = np.asarray(current_stage_joint_positions_deg, dtype=float).copy()
            if limit_push_freeze is None:
                limit_push_freeze = self.limit_push_freeze_payload
        conditioning_result = self.joint_target_conditioner.condition(
            reference_joint_positions_deg=current_stage_joint_positions_deg,
            proposed_joint_targets_deg=solved_joint_targets_deg,
        )
        solved_joint_targets_deg = conditioning_result.conditioned_targets_deg
        solved_model_joint_targets_deg = solved_joint_targets_deg

        self.last_joint_targets_deg = solved_model_joint_targets_deg
        self.last_commanded_stage_joint_targets_deg = np.asarray(solved_joint_targets_deg, dtype=float).copy()

        stage_end_effector_position = self.stage_io.read_stage_end_effector_position(stage) if stage is not None else None
        stage_joint_positions_deg = None
        if stage is not None:
            self.stage_io.write_joint_targets_deg(stage, solved_joint_targets_deg)
            if self.write_joint_state_directly:
                self.stage_io.write_joint_state_deg(stage, solved_joint_targets_deg)
            stage_joint_positions_deg = self.stage_io.read_stage_joint_positions_deg(stage)
            stage_end_effector_position = self.stage_io.read_stage_end_effector_position(stage)

        achieved_pose = self.model.forward_kinematics(solved_model_joint_targets_deg)
        achieved_position = achieved_pose[:3, 3]
        target_position = target_pose[:3, 3]
        position_error = target_position - achieved_position
        target_stage_position = target_stage_pose[:3, 3]
        achieved_stage_position = (self.teleop_mapper.model_to_stage_transform @ achieved_pose)[:3, 3]
        stage_end_effector_error = None
        if stage_end_effector_position is not None:
            stage_end_effector_error = (target_stage_position - stage_end_effector_position).tolist()
        heuristic_safety_advisory = self.heuristic_safety_guard.evaluate(
            mapped_delta=None if map_result.mapped_delta_model is None else map_result.mapped_delta_model,
            stage_end_effector_error=stage_end_effector_error,
            stage_dls_delta_deg=stage_dls_delta_deg,
            joint_names=list(self.model.joint_names),
        )
        lower_limits_deg = self.lower_joint_limits_deg
        upper_limits_deg = self.upper_joint_limits_deg
        joint_limit_safety_advisory = self.joint_limit_safety_guard.evaluate(
            joint_names=list(self.model.joint_names),
            proposed_joint_targets_deg=solved_model_joint_targets_deg,
            lower_limits_deg=lower_limits_deg,
            upper_limits_deg=upper_limits_deg,
        )
        advisory_joint_snapshot = None
        advisory_joint_name = joint_limit_safety_advisory.get("joint_name")
        if advisory_joint_name in self.model.joint_names:
            advisory_idx = self.model.joint_names.index(advisory_joint_name)
            advisory_joint_snapshot = {
                "joint_name": advisory_joint_name,
                "joint_index": advisory_idx,
                "current_joint_deg": None if stage_joint_positions_deg is None else float(stage_joint_positions_deg[advisory_idx]),
                "target_joint_deg": None if advisory_idx >= len(solved_model_joint_targets_deg) else float(solved_model_joint_targets_deg[advisory_idx]),
                "start_joint_deg": None if self.start_stage_joint_positions_deg is None else float(self.start_stage_joint_positions_deg[advisory_idx]),
                "lower_limit_deg": None if advisory_idx >= len(lower_limits_deg) else float(lower_limits_deg[advisory_idx]),
                "upper_limit_deg": None if advisory_idx >= len(upper_limits_deg) else float(upper_limits_deg[advisory_idx]),
            }
        all_advisories = [heuristic_safety_advisory, joint_limit_safety_advisory]
        severity_rank = {"ok": 0, "warn": 1, "critical": 2}
        active_advisory = max(all_advisories, key=lambda item: severity_rank.get(str(item.get("severity", "ok")), 0))
        if limit_push_freeze is not None:
            active_advisory = dict(limit_push_freeze)
            all_advisories = [active_advisory, *all_advisories]
        teleop_safety_advisory = {
            "active": active_advisory,
            "all": all_advisories,
            "joint_limit_snapshot": advisory_joint_snapshot,
        }
        stage_vs_model_joint_delta = None
        if stage_joint_positions_deg is not None:
            stage_vs_model_joint_delta = (stage_joint_positions_deg - solved_model_joint_targets_deg).tolist()
        stage_error_score = self._compute_stage_error_score(stage_end_effector_error)
        result = {
            "timestamp": packet_timestamp,
            "joint_names": self.model.joint_names,
            "joint_targets_deg": solved_joint_targets_deg.tolist(),
            "joint_lower_limits_deg": lower_limits_deg.tolist(),
            "joint_upper_limits_deg": upper_limits_deg.tolist(),
            "stage_dls_delta_deg": None if stage_dls_delta_deg is None else stage_dls_delta_deg.tolist(),
            "stage_dls_unclipped_delta_deg": None if stage_dls_unclipped_delta_deg is None else stage_dls_unclipped_delta_deg.tolist(),
            "stage_dls_raw_position_error": None if stage_dls_raw_position_error is None else stage_dls_raw_position_error.tolist(),
            "stage_dls_clamped_position_error": None if stage_dls_clamped_position_error is None else stage_dls_clamped_position_error.tolist(),
            "stage_error_score": stage_error_score,
            "teleop_safety_advisory": teleop_safety_advisory,
            "stage_dls_joint_weights": None if stage_dls_joint_weights is None else stage_dls_joint_weights.tolist(),
            "joint_target_conditioning": {
                "max_delta_deg_per_tick": self.joint_target_max_delta_deg_per_tick,
                "clipped": conditioning_result.clipped,
                "delta_before_clip_deg": conditioning_result.delta_before_clip_deg.tolist(),
                "delta_after_clip_deg": conditioning_result.delta_after_clip_deg.tolist(),
                "unclipped_joint_targets_deg": conditioning_result.unclipped_targets_deg.tolist(),
                "conditioned_joint_targets_deg": conditioning_result.conditioned_targets_deg.tolist(),
            },
            "output_freeze": limit_push_freeze,
            "joint_control_profile": self.position_only_joint_control_profile if self.position_only else "all_solve_v1",
            "joint_control_modes": joint_control_modes,
            "stage_dls_debug": stage_dls_debug,
            "model_joint_targets_deg": solved_model_joint_targets_deg.tolist(),
            "stage_joint_positions_deg": None if stage_joint_positions_deg is None else stage_joint_positions_deg.tolist(),
            "stage_start_joint_positions_deg": None if self.start_stage_joint_positions_deg is None else self.start_stage_joint_positions_deg.tolist(),
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
            "ik_method": "stage_differential_dls" if stage_dls_delta_deg is not None else "model_iterative_dls",
        }
        hand_position = np.asarray(hand_state.get("position", [0.0, 0.0, 0.0]), dtype=float)
        event_payload = {
            "status": "frozen" if self.limit_push_freeze_active else ("applied" if stage is not None else "solved"),
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
                    "stage_error_score": stage_error_score,
                "output_freeze": limit_push_freeze,
                "teleop_safety_advisory": teleop_safety_advisory,
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
        self._last_playing_state = None
        self._play_state_path = DEFAULT_SIM_PLAY_STATE_PATH

    def _refresh_status_window(self) -> None:
        self._status_ui.update(self.controller, self.controller.last_debug_payload)

    def _write_play_state(self, playing: bool) -> None:
        if self._last_playing_state is playing:
            return
        payload = {"playing": bool(playing), "updated_at": time.time()}
        tmp_path = self._play_state_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload) + "\n")
        tmp_path.replace(self._play_state_path)
        self._last_playing_state = bool(playing)

    def _on_update(self, _event: object) -> None:
        import omni.timeline

        timeline = omni.timeline.get_timeline_interface()
        is_playing = bool(timeline.is_playing()) if timeline is not None else True
        self._write_play_state(is_playing)
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
                if status == "frozen":
                    joint_name = ((debug_payload.get("output_freeze") or {}).get("joint_name") or "?")
                    if self._last_status != "frozen":
                        print(f"Hope Jr teleop: output frozen on {joint_name}; press A to reset")
                    self._last_status = "frozen"
                    self._last_wait_seconds = None
                    return
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
        self._write_play_state(True)
        print(f"Hope Jr IK controller subscribed to Isaac update stream at {self.interval_s:.3f}s interval")
        return self

    def stop(self) -> None:
        self._subscription = None
        self._write_play_state(False)
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
        joint_root_path=args.joint_root_path,
        position_scale=args.position_scale,
        world_offset=np.asarray(args.world_offset, dtype=float),
        world_rotate_xyz_deg=np.asarray(args.world_rotate_xyz, dtype=float),
        quest_position_axes=_parse_position_axes(args.quest_position_axes),
        quest_position_signs=np.asarray(args.quest_position_signs, dtype=float),
        position_only=args.position_only,
        debug_path=Path(args.debug_path),
        use_udp=args.use_udp,
        udp_listen_host=args.udp_listen_host,
        udp_listen_port=args.udp_listen_port,
        teleop_debug_root=args.teleop_debug_root,
        show_teleop_debug=args.show_teleop_debug,
        anchor_delay_s=args.anchor_delay_s,
        grip_threshold=args.grip_threshold,
        event_log_path=Path(args.event_log_path),
        quest_deadband_m=args.quest_deadband_m,
        packet_stale_timeout_s=args.packet_stale_timeout_s,
        end_effector_path=args.end_effector_path,
        write_joint_state_directly=args.write_joint_state_directly,
    )


def start_script_editor_loop(
    *,
    lerobot_repo: str | Path = DEFAULT_LEROBOT_REPO,
    joint_root_path: str | None = None,
    position_scale: float | None = None,
    world_offset: list[float] | tuple[float, float, float] | None = None,
    world_rotate_xyz: list[float] | tuple[float, float, float] | None = None,
    quest_position_axes: str | None = None,
    quest_position_signs: list[float] | tuple[float, float, float] | None = None,
    position_only: bool | None = None,
    debug_path: str | Path | None = None,
    use_udp: bool | None = None,
    udp_listen_host: str | None = None,
    udp_listen_port: int | None = None,
    teleop_debug_root: str | None = None,
    show_teleop_debug: bool | None = None,
    anchor_delay_s: float | None = None,
    grip_threshold: float | None = None,
    event_log_path: str | Path | None = None,
    quest_deadband_m: float | None = None,
    packet_stale_timeout_s: float | None = None,
    end_effector_path: str | None = None,
    write_joint_state_directly: bool | None = None,
    interval_s: float | None = None,
    dry_run: bool | None = None,
    consume_only_new: bool | None = None,
    reset_targets_on_stop: bool | None = None,
    reset_target_value_deg: float | None = None,
) -> HopeJrIsaacUpdateLoop:
    global _ACTIVE_LOOP
    stop_script_editor_loop()
    sim_config = _load_repo_sim_config(lerobot_repo)
    controller_defaults = _get_config_section(sim_config, "controller_defaults")
    script_defaults = _get_config_section(sim_config, "script_editor_test_defaults")
    joint_root_path = joint_root_path if joint_root_path is not None else controller_defaults.get("joint_root_path", DEFAULT_JOINT_ROOT_PATH)
    position_scale = float(position_scale if position_scale is not None else script_defaults.get("position_scale", controller_defaults.get("position_scale", 1.0)))
    world_offset = world_offset if world_offset is not None else controller_defaults.get("world_offset", [0.0, 0.0, 0.0])
    world_rotate_xyz = world_rotate_xyz if world_rotate_xyz is not None else controller_defaults.get("world_rotate_xyz", [0.0, 0.0, 0.0])
    quest_position_axes = quest_position_axes if quest_position_axes is not None else script_defaults.get("quest_position_axes", controller_defaults.get("quest_position_axes", "xyz"))
    quest_position_signs = quest_position_signs if quest_position_signs is not None else script_defaults.get("quest_position_signs", controller_defaults.get("quest_position_signs", [1.0, 1.0, 1.0]))
    position_only = bool(position_only if position_only is not None else script_defaults.get("position_only", True))
    debug_path = debug_path if debug_path is not None else controller_defaults.get("debug_path", str(DEFAULT_DEBUG_PATH))
    use_udp = bool(use_udp if use_udp is not None else script_defaults.get("use_udp", controller_defaults.get("use_udp", True)))
    udp_listen_host = udp_listen_host if udp_listen_host is not None else controller_defaults.get("udp_listen_host", DEFAULT_UDP_LISTEN_HOST)
    udp_listen_port = int(udp_listen_port if udp_listen_port is not None else controller_defaults.get("udp_listen_port", DEFAULT_UDP_LISTEN_PORT))
    teleop_debug_root = teleop_debug_root if teleop_debug_root is not None else controller_defaults.get("teleop_debug_root", DEFAULT_TELEOP_DEBUG_ROOT)
    show_teleop_debug = bool(show_teleop_debug if show_teleop_debug is not None else script_defaults.get("show_teleop_debug", controller_defaults.get("show_teleop_debug", True)))
    anchor_delay_s = float(anchor_delay_s if anchor_delay_s is not None else script_defaults.get("anchor_delay_s", controller_defaults.get("anchor_delay_s", 3.0)))
    grip_threshold = float(grip_threshold if grip_threshold is not None else script_defaults.get("grip_threshold", controller_defaults.get("grip_threshold", 0.25)))
    event_log_path = event_log_path if event_log_path is not None else controller_defaults.get("event_log_path", str(DEFAULT_EVENT_LOG_PATH))
    quest_deadband_m = float(quest_deadband_m if quest_deadband_m is not None else controller_defaults.get("quest_deadband_m", 0.01))
    packet_stale_timeout_s = float(packet_stale_timeout_s if packet_stale_timeout_s is not None else controller_defaults.get("packet_stale_timeout_s", DEFAULT_PACKET_STALE_TIMEOUT_S))
    end_effector_path = end_effector_path if end_effector_path is not None else controller_defaults.get("end_effector_path", DEFAULT_END_EFFECTOR_PATH)
    write_joint_state_directly = bool(write_joint_state_directly if write_joint_state_directly is not None else script_defaults.get("write_joint_state_directly", False))
    interval_s = float(interval_s if interval_s is not None else script_defaults.get("interval_s", 0.05))
    dry_run = bool(dry_run if dry_run is not None else script_defaults.get("dry_run", False))
    consume_only_new = bool(consume_only_new if consume_only_new is not None else script_defaults.get("consume_only_new", True))
    reset_targets_on_stop = bool(reset_targets_on_stop if reset_targets_on_stop is not None else script_defaults.get("reset_targets_on_stop", True))
    reset_target_value_deg = float(reset_target_value_deg if reset_target_value_deg is not None else script_defaults.get("reset_target_value_deg", 0.0))
    latest_debug_path = Path(debug_path)
    latest_event_log_path = Path(event_log_path)
    run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{int((time.time() % 1.0) * 1000):03d}"
    run_debug_path = _build_run_log_path(latest_debug_path, run_id)
    run_event_log_path = _build_run_log_path(latest_event_log_path, run_id)

    controller = HopeJrSimIkController(
        lerobot_repo=Path(lerobot_repo),
        joint_root_path=joint_root_path,
        position_scale=position_scale,
        world_offset=np.asarray(world_offset, dtype=float),
        world_rotate_xyz_deg=np.asarray(world_rotate_xyz, dtype=float),
        quest_position_axes=_parse_position_axes(quest_position_axes),
        quest_position_signs=np.asarray(quest_position_signs, dtype=float),
        position_only=position_only,
        debug_path=run_debug_path,
        use_udp=use_udp,
        udp_listen_host=udp_listen_host,
        udp_listen_port=udp_listen_port,
        teleop_debug_root=teleop_debug_root,
        show_teleop_debug=show_teleop_debug,
        anchor_delay_s=anchor_delay_s,
        grip_threshold=grip_threshold,
        event_log_path=run_event_log_path,
        quest_deadband_m=quest_deadband_m,
        packet_stale_timeout_s=packet_stale_timeout_s,
        end_effector_path=end_effector_path,
        write_joint_state_directly=write_joint_state_directly,
    )
    controller.latest_debug_path = latest_debug_path
    controller.latest_event_log_path = latest_event_log_path
    try:
        controller.latest_event_log_path.unlink(missing_ok=True)
        controller.latest_debug_path.unlink(missing_ok=True)
    except Exception:
        pass
    print(f"Hope Jr IK run log: {controller.event_log_path}")
    print(f"Hope Jr IK run debug: {controller.debug_path}")
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
    config = _load_repo_sim_config(DEFAULT_LEROBOT_REPO)
    controller_defaults = _get_config_section(config, "controller_defaults")
    parser = argparse.ArgumentParser(description="First-pass Hope Jr Quest -> Sim IK controller")
    parser.add_argument("--lerobot-repo", type=Path, default=DEFAULT_LEROBOT_REPO)
    parser.add_argument("--joint-root-path", default=controller_defaults.get("joint_root_path", DEFAULT_JOINT_ROOT_PATH))
    parser.add_argument("--position-scale", type=float, default=float(controller_defaults.get("position_scale", 1.0)))
    parser.add_argument("--world-offset", nargs=3, type=float, default=list(controller_defaults.get("world_offset", [0.0, 0.0, 0.0])))
    parser.add_argument("--world-rotate-xyz", nargs=3, type=float, default=list(controller_defaults.get("world_rotate_xyz", [0.0, 0.0, 0.0])))
    parser.add_argument("--quest-position-axes", default=controller_defaults.get("quest_position_axes", "xyz"))
    parser.add_argument("--quest-position-signs", nargs=3, type=float, default=list(controller_defaults.get("quest_position_signs", [1.0, 1.0, 1.0])))
    parser.add_argument("--position-only", action="store_true")
    parser.add_argument("--debug-path", type=Path, default=Path(controller_defaults.get("debug_path", str(DEFAULT_DEBUG_PATH))))
    parser.add_argument("--use-udp", action="store_true", default=bool(controller_defaults.get("use_udp", True)))
    parser.add_argument("--udp-listen-host", default=controller_defaults.get("udp_listen_host", DEFAULT_UDP_LISTEN_HOST))
    parser.add_argument("--udp-listen-port", type=int, default=int(controller_defaults.get("udp_listen_port", DEFAULT_UDP_LISTEN_PORT)))
    parser.add_argument("--teleop-debug-root", default=controller_defaults.get("teleop_debug_root", DEFAULT_TELEOP_DEBUG_ROOT))
    parser.add_argument("--show-teleop-debug", action="store_true", default=bool(controller_defaults.get("show_teleop_debug", True)))
    parser.add_argument("--anchor-delay-s", type=float, default=float(controller_defaults.get("anchor_delay_s", 3.0)))
    parser.add_argument("--grip-threshold", type=float, default=float(controller_defaults.get("grip_threshold", 0.25)))
    parser.add_argument("--event-log-path", type=Path, default=Path(controller_defaults.get("event_log_path", str(DEFAULT_EVENT_LOG_PATH))))
    parser.add_argument("--quest-deadband-m", type=float, default=float(controller_defaults.get("quest_deadband_m", 0.01)))
    parser.add_argument("--packet-stale-timeout-s", type=float, default=float(controller_defaults.get("packet_stale_timeout_s", DEFAULT_PACKET_STALE_TIMEOUT_S)))
    parser.add_argument("--end-effector-path", default=controller_defaults.get("end_effector_path", DEFAULT_END_EFFECTOR_PATH))
    parser.add_argument("--write-joint-state-directly", action="store_true")
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
        print(f"watching udp://{args.udp_listen_host}:{args.udp_listen_port}")
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
