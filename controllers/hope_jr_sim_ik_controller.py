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
from controllers.ik_weight_profiles import DEFAULT_STAGE_WEIGHT_PROFILE, build_stage_joint_weights, list_stage_weight_profiles
from controllers.joint_control_policy import (
    DEFAULT_POSITION_ONLY_JOINT_CONTROL_PROFILE,
    JOINT_CONTROL_HOLD_LAST,
    JOINT_CONTROL_HOLD_START,
    build_joint_control_modes,
)
from controllers.teleop_packet_source import TeleopPacketSource
from controllers.teleop_safety_advisor import DEFAULT_TELEOP_ADVISOR_PROFILE, TeleopSafetyAdvisor
from controllers.stage_io import HopeJrStageIo

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
DEFAULT_SIM_PLAY_STATE_PATH = Path("/tmp/hope_jr_sim_play_state.json")
DEFAULT_STOP_TARGETS_DEG = {}
DEFAULT_MODEL_JOINT_SIGNS = {
    "right_shoulder_pitch": -1.0,
    "right_shoulder_yaw": -1.0,
    "right_arm_twist": -1.0,
    "right_elbow": -1.0,
    "right_forearm_twist": 1.0,
    "right_wrist": 1.0,
    "right_palm": 1.0,
}
DEFAULT_STAGE_DLS_LAMBDA = 0.01
DEFAULT_STAGE_DLS_MAX_STEP_DEG = 8.0
DEFAULT_STAGE_TASK_DELTA_CLAMP_M = 0.03
DEFAULT_STAGE_ERROR_SCORE_WINDOW = 120
DEFAULT_STAGE_POSITION_ONLY_WEIGHT_OVERRIDES = {
    "right_forearm_twist": 0.0,
    "right_wrist": 0.0,
    "right_palm": 0.0,
}

_ACTIVE_LOOP = None


def _sanitize_profile_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value))


def _profile_specific_log_path(base_path: Path, profile_name: str) -> Path:
    sanitized = _sanitize_profile_name(profile_name)
    if base_path == DEFAULT_DEBUG_PATH:
        return base_path.with_name(f"{base_path.stem}_{sanitized}{base_path.suffix}")
    if base_path == DEFAULT_EVENT_LOG_PATH:
        return base_path.with_name(f"{base_path.stem}_{sanitized}{base_path.suffix}")
    return base_path


def _resolve_profile_log_paths(debug_path: Path, event_log_path: Path, profile_name: str) -> tuple[Path, Path]:
    return _profile_specific_log_path(debug_path, profile_name), _profile_specific_log_path(event_log_path, profile_name)


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
        stage_weight_profile: str = DEFAULT_STAGE_WEIGHT_PROFILE,
    ):
        self.lerobot_repo = lerobot_repo
        self.packet_path = packet_path
        self.joint_root_path = joint_root_path.rstrip("/")
        self.articulation_root_path = self.joint_root_path.rsplit("/", 1)[0]
        self.position_only = position_only
        self.debug_path = debug_path

        self.teleop_debug_root = teleop_debug_root.rstrip("/")
        self.show_teleop_debug = show_teleop_debug
        self.anchor_delay_s = anchor_delay_s
        self.event_log_path = event_log_path
        self.packet_stale_timeout_s = float(packet_stale_timeout_s)
        self.end_effector_path = end_effector_path
        self.write_joint_state_directly = bool(write_joint_state_directly)
        self.stage_weight_profile = str(stage_weight_profile)
        self.teleop_safety_advisor = TeleopSafetyAdvisor(DEFAULT_TELEOP_ADVISOR_PROFILE)
        self.packet_source = TeleopPacketSource(
            packet_path=packet_path,
            use_udp=use_udp,
            udp_listen_host=udp_listen_host,
            udp_listen_port=udp_listen_port,
        )
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
        self.stage_joint_weights = np.asarray(
            build_stage_joint_weights(self.model.joint_names, self.stage_weight_profile),
            dtype=float,
        )
        self.position_only_joint_control_profile = DEFAULT_POSITION_ONLY_JOINT_CONTROL_PROFILE
        self.position_only_joint_control_modes = build_joint_control_modes(
            self.model.joint_names, self.position_only_joint_control_profile
        )
        self.last_joint_targets_deg = np.zeros(len(self.model.joint_names), dtype=float)
        self.last_commanded_stage_joint_targets_deg = None
        self.neutral_model_joint_targets_deg = np.array([DEFAULT_STOP_TARGETS_DEG.get(name, 0.0) for name in self.model.joint_names], dtype=float)
        self.start_stage_joint_positions_deg = None
        self.last_packet_timestamp = None
        self.stage_error_norm_window = deque(maxlen=DEFAULT_STAGE_ERROR_SCORE_WINDOW)
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
        self.stage_io = HopeJrStageIo(
            articulation_root_path=self.articulation_root_path,
            joint_root_path=self.joint_root_path,
            end_effector_path=self.end_effector_path,
            joint_names=self.model.joint_names,
            model_joint_signs=self.model_joint_signs,
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
        stage = self.stage_io.get_stage()
        if stage is None:
            return
        target_values = np.array(
            [DEFAULT_STOP_TARGETS_DEG.get(joint_name, float(target_value_deg)) for joint_name in self.model.joint_names],
            dtype=float,
        )
        self.stage_io.write_joint_targets_deg(stage, target_values)
        if reset_joint_state:
            self.stage_io.write_joint_state_deg(stage, target_values)
        self.last_joint_targets_deg = self.stage_io.stage_to_model_joint_positions_deg(target_values)
        self.last_commanded_stage_joint_targets_deg = np.asarray(target_values, dtype=float).copy()
        self.start_stage_joint_positions_deg = np.asarray(target_values, dtype=float).copy()

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
        position_error = self._clip_vector_norm(raw_position_error, DEFAULT_STAGE_TASK_DELTA_CLAMP_M)
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
                override = DEFAULT_STAGE_POSITION_ONLY_WEIGHT_OVERRIDES.get(name)
                if override is not None:
                    solve_joint_weights[index] = min(solve_joint_weights[index], float(override))
        task_jacobian = task_jacobian * solve_joint_weights[None, :]

        damping = DEFAULT_STAGE_DLS_LAMBDA
        try:
            delta_rad = task_jacobian.T @ np.linalg.solve(
                task_jacobian @ task_jacobian.T + (damping**2) * np.eye(task_jacobian.shape[0]),
                task_error,
            )
        except np.linalg.LinAlgError:
            return None

        delta_deg = np.rad2deg(delta_rad) * solve_joint_weights
        unclipped_delta_deg = delta_deg.copy()
        delta_deg = np.clip(delta_deg, -DEFAULT_STAGE_DLS_MAX_STEP_DEG, DEFAULT_STAGE_DLS_MAX_STEP_DEG)
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
        solved_model_joint_targets_deg = self.stage_io.stage_to_model_joint_positions_deg(solved_joint_targets_deg)

        self.last_joint_targets_deg = solved_model_joint_targets_deg
        self.last_commanded_stage_joint_targets_deg = np.asarray(solved_joint_targets_deg, dtype=float).copy()

        stage_end_effector_position = self.stage_io.read_stage_end_effector_position(stage) if stage is not None else None
        stage_joint_positions_deg = None
        stage_model_joint_positions_deg = None
        if stage is not None:
            self.stage_io.write_joint_targets_deg(stage, solved_joint_targets_deg)
            if self.write_joint_state_directly:
                self.stage_io.write_joint_state_deg(stage, solved_joint_targets_deg)
            stage_joint_positions_deg = self.stage_io.read_stage_joint_positions_deg(stage)
            if stage_joint_positions_deg is not None:
                stage_model_joint_positions_deg = self.stage_io.stage_to_model_joint_positions_deg(stage_joint_positions_deg)
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
        teleop_safety_advisory = self.teleop_safety_advisor.evaluate(
            mapped_delta=None if map_result.mapped_delta_model is None else map_result.mapped_delta_model,
            stage_end_effector_error=stage_end_effector_error,
            stage_dls_delta_deg=stage_dls_delta_deg,
            joint_names=list(self.model.joint_names),
        )
        stage_vs_model_joint_delta = None
        if stage_model_joint_positions_deg is not None:
            stage_vs_model_joint_delta = (stage_model_joint_positions_deg - solved_model_joint_targets_deg).tolist()
        stage_error_score = self._compute_stage_error_score(stage_end_effector_error)
        result = {
            "timestamp": packet_timestamp,
            "joint_names": self.model.joint_names,
            "joint_targets_deg": solved_joint_targets_deg.tolist(),
            "stage_dls_delta_deg": None if stage_dls_delta_deg is None else stage_dls_delta_deg.tolist(),
            "stage_dls_unclipped_delta_deg": None if stage_dls_unclipped_delta_deg is None else stage_dls_unclipped_delta_deg.tolist(),
            "stage_dls_raw_position_error": None if stage_dls_raw_position_error is None else stage_dls_raw_position_error.tolist(),
            "stage_dls_clamped_position_error": None if stage_dls_clamped_position_error is None else stage_dls_clamped_position_error.tolist(),
            "stage_weight_profile": self.stage_weight_profile,
            "stage_error_score": stage_error_score,
            "teleop_safety_advisory": teleop_safety_advisory,
            "stage_dls_joint_weights": None if stage_dls_joint_weights is None else stage_dls_joint_weights.tolist(),
            "joint_control_profile": self.position_only_joint_control_profile if self.position_only else "all_solve_v1",
            "joint_control_modes": joint_control_modes,
            "stage_dls_debug": stage_dls_debug,
            "model_joint_targets_deg": solved_model_joint_targets_deg.tolist(),
            "stage_joint_positions_deg": None if stage_joint_positions_deg is None else stage_joint_positions_deg.tolist(),
            "stage_start_joint_positions_deg": None if self.start_stage_joint_positions_deg is None else self.start_stage_joint_positions_deg.tolist(),
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
            "ik_method": "stage_differential_dls" if stage_dls_delta_deg is not None else "model_iterative_dls",
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
                "stage_weight_profile": self.stage_weight_profile,
                "stage_error_score": stage_error_score,
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
    debug_path, event_log_path = _resolve_profile_log_paths(Path(args.debug_path), Path(args.event_log_path), args.stage_weight_profile)
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
        debug_path=debug_path,
        use_udp=args.use_udp,
        udp_listen_host=args.udp_listen_host,
        udp_listen_port=args.udp_listen_port,
        teleop_debug_root=args.teleop_debug_root,
        show_teleop_debug=args.show_teleop_debug,
        anchor_delay_s=args.anchor_delay_s,
        grip_threshold=args.grip_threshold,
        event_log_path=event_log_path,
        quest_deadband_m=args.quest_deadband_m,
        packet_stale_timeout_s=args.packet_stale_timeout_s,
        end_effector_path=args.end_effector_path,
        write_joint_state_directly=args.write_joint_state_directly,
        stage_weight_profile=args.stage_weight_profile,
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
    stage_weight_profile: str = DEFAULT_STAGE_WEIGHT_PROFILE,
    interval_s: float = 0.05,
    dry_run: bool = False,
    consume_only_new: bool = True,
    reset_targets_on_stop: bool = True,
    reset_target_value_deg: float = 0.0,
) -> HopeJrIsaacUpdateLoop:
    global _ACTIVE_LOOP
    stop_script_editor_loop()
    resolved_debug_path, resolved_event_log_path = _resolve_profile_log_paths(Path(debug_path), Path(event_log_path), stage_weight_profile)
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
        debug_path=resolved_debug_path,
        use_udp=use_udp,
        udp_listen_host=udp_listen_host,
        udp_listen_port=udp_listen_port,
        teleop_debug_root=teleop_debug_root,
        show_teleop_debug=show_teleop_debug,
        anchor_delay_s=anchor_delay_s,
        grip_threshold=grip_threshold,
        event_log_path=resolved_event_log_path,
        quest_deadband_m=quest_deadband_m,
        packet_stale_timeout_s=packet_stale_timeout_s,
        end_effector_path=end_effector_path,
        write_joint_state_directly=write_joint_state_directly,
        stage_weight_profile=stage_weight_profile,
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
    parser.add_argument("--stage-weight-profile", choices=list_stage_weight_profiles(), default=DEFAULT_STAGE_WEIGHT_PROFILE)
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
