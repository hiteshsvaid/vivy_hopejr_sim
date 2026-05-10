#!/usr/bin/env python3

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class QuestTeleopMapperResult:
    target_pose: np.ndarray | None
    hand_state: dict[str, Any]
    waiting_for_anchor: bool
    follow_target_enabled: bool
    quest_current_position: np.ndarray
    quest_anchor_position: np.ndarray | None
    quest_mapped_position_stage: np.ndarray
    sim_target_position_stage: np.ndarray
    quest_delta: np.ndarray | None
    mapped_delta_model: np.ndarray | None
    anchor_captured_payload: dict[str, Any] | None = None


class QuestTeleopMapper:
    def __init__(
        self,
        *,
        position_scale: float,
        world_offset: np.ndarray,
        world_rotate_xyz_deg: np.ndarray,
        quest_position_axes: tuple[int, int, int],
        quest_position_signs: np.ndarray,
        position_only: bool,
        anchor_delay_s: float,
        quest_deadband_m: float,
        make_pose,
        hand_key: str = "right_hand",
    ):
        self.position_scale = float(position_scale)
        self.world_offset = np.asarray(world_offset, dtype=float)
        self.world_rotation = Rotation.from_euler("XYZ", world_rotate_xyz_deg, degrees=True).as_matrix()
        self.quest_position_axes = tuple(quest_position_axes)
        self.quest_position_signs = np.asarray(quest_position_signs, dtype=float)
        self.position_only = bool(position_only)
        self.anchor_delay_s = float(anchor_delay_s)
        self.quest_deadband_m = float(quest_deadband_m)
        self.make_pose = make_pose
        self.hand_key = str(hand_key)
        self._thumbstick_click_last = False
        self._follow_target_enabled = False
        self.reset()

    @staticmethod
    def _normalize_hand_state(hand: dict[str, Any]) -> dict[str, Any]:
        normalized_hand = dict(hand)
        if "position" not in normalized_hand and "pos" in normalized_hand:
            normalized_hand["position"] = normalized_hand["pos"]
        if "orientation_xyzw" not in normalized_hand and "rot" in normalized_hand:
            normalized_hand["orientation_xyzw"] = normalized_hand["rot"]
        return normalized_hand

    def reset(self) -> None:
        self.quest_anchor_position = None
        self.quest_anchor_rotation = None
        self.sim_anchor_pose = None
        self.stage_anchor_pose = None
        self.model_to_stage_transform = np.eye(4)
        self.stage_to_model_transform = np.eye(4)
        self.anchor_ready_time = time.time() + self.anchor_delay_s
        self._thumbstick_click_last = False
        self._follow_target_enabled = False

    def _handle_thumbstick_click_action(
        self,
        hand: dict[str, Any],
        *,
        quest_position: np.ndarray,
        quest_rotation: np.ndarray,
        current_sim_pose: np.ndarray,
        current_stage_pose: np.ndarray | None,
        anchor_joint_targets_deg: np.ndarray,
    ) -> tuple[bool, dict[str, Any] | None]:
        thumbstick_click = bool(hand.get("thumbstick_click", False))
        click_rising = thumbstick_click and not self._thumbstick_click_last
        self._thumbstick_click_last = thumbstick_click
        if not click_rising:
            return False, None

        if self._follow_target_enabled:
            self._follow_target_enabled = False
            return True, None

        if not hand.get("enabled", True) or hand.get("clutch", False):
            return True, None

        waiting_pose = current_stage_pose if current_stage_pose is not None else current_sim_pose
        self.quest_anchor_position = quest_position.copy()
        self.quest_anchor_rotation = quest_rotation.copy()
        self.sim_anchor_pose = current_sim_pose.copy()
        self.stage_anchor_pose = waiting_pose.copy()
        try:
            self.model_to_stage_transform = self.stage_anchor_pose @ np.linalg.inv(self.sim_anchor_pose)
            self.stage_to_model_transform = np.linalg.inv(self.model_to_stage_transform)
        except np.linalg.LinAlgError:
            self.model_to_stage_transform = np.eye(4)
            self.stage_to_model_transform = np.eye(4)
        self._follow_target_enabled = True
        anchor_payload = {
            "status": "anchor_captured",
            "quest_anchor_position": self.quest_anchor_position.tolist(),
            "sim_anchor_position": self.sim_anchor_pose[:3, 3].tolist(),
            "stage_anchor_position": self.stage_anchor_pose[:3, 3].tolist(),
            "anchor_joint_targets_deg": np.asarray(anchor_joint_targets_deg, dtype=float).tolist(),
        }
        return True, anchor_payload

    def map_packet(
        self,
        packet: dict[str, Any],
        *,
        current_sim_pose: np.ndarray,
        current_stage_pose: np.ndarray | None,
        anchor_joint_targets_deg: np.ndarray,
    ) -> QuestTeleopMapperResult | None:
        normalized = packet.get("normalized")
        if not isinstance(normalized, dict):
            return None
        hand = normalized.get(self.hand_key)
        if not isinstance(hand, dict):
            return None
        hand = self._normalize_hand_state(hand)
        if not hand.get("enabled", True) or hand.get("clutch", False):
            return None

        position = hand.get("position")
        orientation_xyzw = hand.get("orientation_xyzw")
        if position is None or orientation_xyzw is None:
            return None

        quest_position = np.asarray(position, dtype=float)
        quest_rotation = Rotation.from_quat(np.asarray(orientation_xyzw, dtype=float)).as_matrix()
        click_handled, anchor_payload = self._handle_thumbstick_click_action(
            hand,
            quest_position=quest_position,
            quest_rotation=quest_rotation,
            current_sim_pose=current_sim_pose,
            current_stage_pose=current_stage_pose,
            anchor_joint_targets_deg=anchor_joint_targets_deg,
        )
        if click_handled and not self._follow_target_enabled:
            return QuestTeleopMapperResult(
                target_pose=None,
                hand_state=hand,
                waiting_for_anchor=True,
                follow_target_enabled=False,
                quest_current_position=quest_position,
                quest_anchor_position=quest_position,
                quest_mapped_position_stage=current_stage_pose[:3, 3].copy() if current_stage_pose is not None else current_sim_pose[:3, 3].copy(),
                sim_target_position_stage=current_stage_pose[:3, 3].copy() if current_stage_pose is not None else current_sim_pose[:3, 3].copy(),
                quest_delta=None,
                mapped_delta_model=None,
                anchor_captured_payload=None,
            )
        if not self._follow_target_enabled:
            return None

        if self.quest_anchor_position is None or self.sim_anchor_pose is None or self.stage_anchor_pose is None:
            waiting_pose = current_stage_pose if current_stage_pose is not None else current_sim_pose
            if time.time() < self.anchor_ready_time:
                return QuestTeleopMapperResult(
                    target_pose=None,
                    hand_state=hand,
                    waiting_for_anchor=True,
                    follow_target_enabled=self._follow_target_enabled,
                    quest_current_position=quest_position,
                    quest_anchor_position=quest_position,
                    quest_mapped_position_stage=waiting_pose[:3, 3].copy(),
                    sim_target_position_stage=waiting_pose[:3, 3].copy(),
                    quest_delta=None,
                    mapped_delta_model=None,
                )
            self.quest_anchor_position = quest_position.copy()
            self.quest_anchor_rotation = quest_rotation.copy()
            self.sim_anchor_pose = current_sim_pose.copy()
            self.stage_anchor_pose = waiting_pose.copy()
            try:
                self.model_to_stage_transform = self.stage_anchor_pose @ np.linalg.inv(self.sim_anchor_pose)
                self.stage_to_model_transform = np.linalg.inv(self.model_to_stage_transform)
            except np.linalg.LinAlgError:
                self.model_to_stage_transform = np.eye(4)
                self.stage_to_model_transform = np.eye(4)
            anchor_payload = {
                "status": "anchor_captured",
                "quest_anchor_position": self.quest_anchor_position.tolist(),
                "sim_anchor_position": self.sim_anchor_pose[:3, 3].tolist(),
                "stage_anchor_position": self.stage_anchor_pose[:3, 3].tolist(),
                "anchor_joint_targets_deg": np.asarray(anchor_joint_targets_deg, dtype=float).tolist(),
            }
        else:
            anchor_payload = None

        quest_delta = quest_position - self.quest_anchor_position
        if self.quest_deadband_m > 0.0:
            small = np.abs(quest_delta) < self.quest_deadband_m
            quest_delta = np.where(small, 0.0, quest_delta)
        remapped_delta = quest_delta[list(self.quest_position_axes)] * self.quest_position_signs
        position_delta_stage = self.world_rotation @ (remapped_delta * self.position_scale)
        quest_mapped_position_stage = self.stage_anchor_pose[:3, 3] + position_delta_stage
        desired_target_position_stage = quest_mapped_position_stage + self.world_offset
        if self.position_only:
            target_rotation_stage = self.stage_anchor_pose[:3, :3]
        else:
            relative_rotation = quest_rotation @ self.quest_anchor_rotation.T
            target_rotation_stage = self.world_rotation @ relative_rotation @ self.stage_anchor_pose[:3, :3]
        target_pose_stage = self.make_pose(position=desired_target_position_stage, rotation_matrix=target_rotation_stage)
        target_pose = self.stage_to_model_transform @ target_pose_stage
        return QuestTeleopMapperResult(
            target_pose=target_pose,
            hand_state=hand,
            waiting_for_anchor=False,
            follow_target_enabled=self._follow_target_enabled,
            quest_current_position=quest_position,
            quest_anchor_position=self.quest_anchor_position.copy(),
            quest_mapped_position_stage=quest_mapped_position_stage,
            sim_target_position_stage=desired_target_position_stage,
            quest_delta=quest_delta,
            mapped_delta_model=target_pose[:3, 3] - self.sim_anchor_pose[:3, 3],
            anchor_captured_payload=anchor_payload,
        )
