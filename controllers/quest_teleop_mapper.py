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
        grip_threshold: float,
        quest_deadband_m: float,
        make_pose,
    ):
        self.position_scale = float(position_scale)
        self.world_offset = np.asarray(world_offset, dtype=float)
        self.world_rotation = Rotation.from_euler("XYZ", world_rotate_xyz_deg, degrees=True).as_matrix()
        self.quest_position_axes = tuple(quest_position_axes)
        self.quest_position_signs = np.asarray(quest_position_signs, dtype=float)
        self.position_only = bool(position_only)
        self.anchor_delay_s = float(anchor_delay_s)
        self.grip_threshold = float(grip_threshold)
        self.quest_deadband_m = float(quest_deadband_m)
        self.make_pose = make_pose
        self.reset()

    def reset(self) -> None:
        self.quest_anchor_position = None
        self.quest_anchor_rotation = None
        self.sim_anchor_pose = None
        self.stage_anchor_pose = None
        self.model_to_stage_transform = np.eye(4)
        self.stage_to_model_transform = np.eye(4)
        self.anchor_ready_time = time.time() + self.anchor_delay_s

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
        hand = normalized.get("right_hand")
        if not isinstance(hand, dict):
            return None
        if not hand.get("enabled", True) or hand.get("clutch", False):
            return None
        if float(hand.get("grip", 0.0)) < self.grip_threshold:
            return None

        quest_position = np.asarray(hand["position"], dtype=float)
        quest_rotation = Rotation.from_quat(np.asarray(hand["orientation_xyzw"], dtype=float)).as_matrix()

        if self.quest_anchor_position is None or self.sim_anchor_pose is None or self.stage_anchor_pose is None:
            waiting_pose = current_stage_pose if current_stage_pose is not None else current_sim_pose
            if time.time() < self.anchor_ready_time:
                return QuestTeleopMapperResult(
                    target_pose=None,
                    hand_state=hand,
                    waiting_for_anchor=True,
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
            self.stage_anchor_pose = current_stage_pose.copy() if current_stage_pose is not None else current_sim_pose.copy()
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
                "anchor_joint_targets_deg": anchor_joint_targets_deg.tolist(),
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
            quest_current_position=quest_position,
            quest_anchor_position=self.quest_anchor_position.copy(),
            quest_mapped_position_stage=quest_mapped_position_stage,
            sim_target_position_stage=desired_target_position_stage,
            quest_delta=quest_delta,
            mapped_delta_model=target_pose[:3, 3] - self.sim_anchor_pose[:3, 3],
            anchor_captured_payload=anchor_payload,
        )
