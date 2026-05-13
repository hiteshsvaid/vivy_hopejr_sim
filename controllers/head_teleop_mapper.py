
#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class HeadTeleopMapperResult:
    head_state: dict[str, Any]
    tracked: bool
    waiting_for_anchor: bool
    tracking_lost: bool
    head_current_degrees: np.ndarray
    head_anchor_degrees: np.ndarray | None
    target_joint_targets_deg: np.ndarray
    anchor_captured_payload: dict[str, Any] | None = None


class HeadTeleopMapper:
    def __init__(
        self,
        *,
        head_joint_names: tuple[str, str],
        neutral_joint_targets_deg: np.ndarray,
        lower_joint_limits_deg: np.ndarray,
        upper_joint_limits_deg: np.ndarray,
        pan_input_clamp_deg: float = 60.0,
        tilt_input_clamp_deg: float = 30.0,
        max_delta_deg_per_tick: np.ndarray | float = 2.0,
    ):
        if len(head_joint_names) != 2:
            raise ValueError(f"Expected head pan/tilt joint names, got {head_joint_names!r}")
        self.head_joint_names = tuple(str(name) for name in head_joint_names)
        self.neutral_joint_targets_deg = np.asarray(neutral_joint_targets_deg, dtype=float)
        self.lower_joint_limits_deg = np.asarray(lower_joint_limits_deg, dtype=float)
        self.upper_joint_limits_deg = np.asarray(upper_joint_limits_deg, dtype=float)
        self.pan_input_clamp_deg = float(pan_input_clamp_deg)
        self.tilt_input_clamp_deg = float(tilt_input_clamp_deg)
        self.max_delta_deg_per_tick = np.asarray(max_delta_deg_per_tick, dtype=float)
        if self.neutral_joint_targets_deg.shape != (2,):
            raise ValueError(f"Expected two neutral head targets, got shape {self.neutral_joint_targets_deg.shape}")
        if self.lower_joint_limits_deg.shape != (2,) or self.upper_joint_limits_deg.shape != (2,):
            raise ValueError("Expected two head joint limits")
        if self.max_delta_deg_per_tick.shape == ():
            self.max_delta_deg_per_tick = np.asarray([float(self.max_delta_deg_per_tick)] * 2, dtype=float)
        elif self.max_delta_deg_per_tick.shape != (2,):
            raise ValueError("Expected head max delta to be scalar or length-2 array")
        self.reset()

    @staticmethod
    def _normalize_head_state(head: dict[str, Any]) -> dict[str, Any]:
        normalized_head = dict(head)
        if "position" not in normalized_head and "pos" in normalized_head:
            normalized_head["position"] = normalized_head["pos"]
        if "orientation_xyzw" not in normalized_head and "rot" in normalized_head:
            normalized_head["orientation_xyzw"] = normalized_head["rot"]
        return normalized_head

    def reset(self) -> None:
        self._head_anchor_degrees: np.ndarray | None = None
        self._tracking_lost = False
        self._current_joint_targets_deg = self.neutral_joint_targets_deg.copy()
        self._previous_joint_targets_deg = self.neutral_joint_targets_deg.copy()
        self._last_head_state: dict[str, Any] = {}

    def _capture_anchor(self, head_degrees: np.ndarray) -> dict[str, Any]:
        self._head_anchor_degrees = np.asarray(head_degrees, dtype=float).copy()
        self._tracking_lost = False
        return {
            "status": "head_anchor_captured",
            "head_anchor_pan_degrees": float(self._head_anchor_degrees[0]),
            "head_anchor_tilt_degrees": float(self._head_anchor_degrees[1]),
            "head_target_joint_targets_deg": self._current_joint_targets_deg.tolist(),
        }

    def map_packet(self, packet: dict[str, Any]) -> HeadTeleopMapperResult | None:
        if not isinstance(packet, dict):
            return None
        head = packet.get("head")
        if not isinstance(head, dict):
            return None
        head = self._normalize_head_state(head)
        tracked = bool(head.get("is_tracked", False))
        pan_degrees = float(head.get("pan_degrees", 0.0))
        tilt_degrees = float(head.get("tilt_degrees", 0.0))
        head_current_degrees = np.asarray([pan_degrees, tilt_degrees], dtype=float)
        self._last_head_state = dict(head)

        if not tracked:
            self._tracking_lost = True
            return HeadTeleopMapperResult(
                head_state=dict(head),
                tracked=False,
                waiting_for_anchor=False,
                tracking_lost=True,
                head_current_degrees=head_current_degrees,
                head_anchor_degrees=None if self._head_anchor_degrees is None else self._head_anchor_degrees.copy(),
                target_joint_targets_deg=self._current_joint_targets_deg.copy(),
            )

        anchor_captured_payload = None
        if self._head_anchor_degrees is None or self._tracking_lost:
            anchor_captured_payload = self._capture_anchor(head_current_degrees)

        assert self._head_anchor_degrees is not None
        head_delta = head_current_degrees - self._head_anchor_degrees
        head_delta = np.asarray(
            [
                float(np.clip(head_delta[0], -self.pan_input_clamp_deg, self.pan_input_clamp_deg)),
                float(np.clip(head_delta[1], -self.tilt_input_clamp_deg, self.tilt_input_clamp_deg)),
            ],
            dtype=float,
        )
        desired_joint_targets_deg = self.neutral_joint_targets_deg + head_delta
        desired_joint_targets_deg = np.clip(desired_joint_targets_deg, self.lower_joint_limits_deg, self.upper_joint_limits_deg)
        delta_from_current = desired_joint_targets_deg - self._current_joint_targets_deg
        delta_from_current = np.clip(delta_from_current, -self.max_delta_deg_per_tick, self.max_delta_deg_per_tick)
        target_joint_targets_deg = self._current_joint_targets_deg + delta_from_current
        self._previous_joint_targets_deg = self._current_joint_targets_deg.copy()
        self._current_joint_targets_deg = np.asarray(target_joint_targets_deg, dtype=float)

        return HeadTeleopMapperResult(
            head_state=dict(head),
            tracked=True,
            waiting_for_anchor=False,
            tracking_lost=False,
            head_current_degrees=head_current_degrees,
            head_anchor_degrees=self._head_anchor_degrees.copy(),
            target_joint_targets_deg=self._current_joint_targets_deg.copy(),
            anchor_captured_payload=anchor_captured_payload,
        )
