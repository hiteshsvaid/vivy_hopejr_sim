#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JointTargetConditioningResult:
    conditioned_targets_deg: np.ndarray
    unclipped_targets_deg: np.ndarray
    delta_before_clip_deg: np.ndarray
    delta_after_clip_deg: np.ndarray
    clipped: bool


class JointTargetConditioner:
    def __init__(self, *, max_delta_deg_per_tick: float | None):
        self.max_delta_deg_per_tick = None if max_delta_deg_per_tick is None else float(max_delta_deg_per_tick)

    def condition(
        self,
        *,
        reference_joint_positions_deg: np.ndarray | None,
        proposed_joint_targets_deg: np.ndarray,
    ) -> JointTargetConditioningResult:
        proposed = np.asarray(proposed_joint_targets_deg, dtype=float).copy()
        if reference_joint_positions_deg is None or self.max_delta_deg_per_tick is None or self.max_delta_deg_per_tick <= 0.0:
            zeros = np.zeros_like(proposed)
            return JointTargetConditioningResult(
                conditioned_targets_deg=proposed.copy(),
                unclipped_targets_deg=proposed.copy(),
                delta_before_clip_deg=zeros.copy(),
                delta_after_clip_deg=zeros.copy(),
                clipped=False,
            )

        reference = np.asarray(reference_joint_positions_deg, dtype=float)
        delta_before = proposed - reference
        delta_after = np.clip(delta_before, -self.max_delta_deg_per_tick, self.max_delta_deg_per_tick)
        conditioned = reference + delta_after
        clipped = bool(np.any(np.abs(delta_after - delta_before) > 1e-9))
        return JointTargetConditioningResult(
            conditioned_targets_deg=conditioned,
            unclipped_targets_deg=proposed,
            delta_before_clip_deg=delta_before,
            delta_after_clip_deg=delta_after,
            clipped=clipped,
        )
