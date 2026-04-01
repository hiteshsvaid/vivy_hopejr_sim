from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class JointLimitSafetyGuardConfig:
    profile_name: str
    soft_margin_deg: float
    warn_margin_deg: float
    critical_margin_deg: float


GUARD_PROFILES: dict[str, JointLimitSafetyGuardConfig] = {
    "joint_limit_guard_v1": JointLimitSafetyGuardConfig(
        profile_name="joint_limit_guard_v1",
        soft_margin_deg=3.0,
        warn_margin_deg=3.0,
        critical_margin_deg=1.0,
    ),
}

DEFAULT_JOINT_LIMIT_GUARD_PROFILE = "joint_limit_guard_v1"


class JointLimitSafetyGuard:
    def __init__(self, profile_name: str = DEFAULT_JOINT_LIMIT_GUARD_PROFILE) -> None:
        if profile_name not in GUARD_PROFILES:
            raise ValueError(
                f"Unknown joint limit safety guard profile '{profile_name}'. Available: {sorted(GUARD_PROFILES)}"
            )
        self.config = GUARD_PROFILES[profile_name]

    def evaluate(
        self,
        *,
        joint_names: list[str],
        proposed_joint_targets_deg: np.ndarray | list[float] | tuple[float, ...] | None,
        lower_limits_deg: np.ndarray | list[float] | tuple[float, ...] | None,
        upper_limits_deg: np.ndarray | list[float] | tuple[float, ...] | None,
    ) -> dict[str, Any]:
        if proposed_joint_targets_deg is None or lower_limits_deg is None or upper_limits_deg is None:
            return {
                "source": "joint_limit",
                "source_label": "Joint-limit guard",
                "profile": self.config.profile_name,
                "severity": "ok",
                "reasons": [],
                "recommendations": [],
                "joint_name": None,
                "joint_index": None,
                "target_joint_deg": None,
                "lower_limit_deg": None,
                "upper_limit_deg": None,
                "lower_margin_deg": None,
                "upper_margin_deg": None,
                "margin_to_limit_deg": None,
                "would_clamp": False,
                "per_joint_margins_deg": {},
            }

        targets = np.asarray(proposed_joint_targets_deg, dtype=float)
        lowers = np.asarray(lower_limits_deg, dtype=float)
        uppers = np.asarray(upper_limits_deg, dtype=float)

        lower_margins = targets - lowers
        upper_margins = uppers - targets
        nearest_margins = np.minimum(lower_margins, upper_margins)
        idx = int(np.argmin(nearest_margins)) if nearest_margins.size else 0
        nearest_margin = float(nearest_margins[idx]) if nearest_margins.size else None
        joint_name = joint_names[idx] if idx < len(joint_names) else None
        target_joint_deg = float(targets[idx]) if idx < len(targets) else None
        lower_limit_deg = float(lowers[idx]) if idx < len(lowers) else None
        upper_limit_deg = float(uppers[idx]) if idx < len(uppers) else None
        lower_margin_deg = float(lower_margins[idx]) if idx < len(lower_margins) else None
        upper_margin_deg = float(upper_margins[idx]) if idx < len(upper_margins) else None

        reasons: list[str] = []
        recommendations: list[str] = []
        severity = "ok"
        would_clamp = False

        if nearest_margin is not None:
            if nearest_margin <= 0.0:
                severity = "critical"
                would_clamp = True
                reasons.append("Joint target exceeds a configured limit")
                recommendations.extend(["Clamp to soft joint limit", "Freeze or re-anchor if this repeats"])
            elif nearest_margin <= self.config.critical_margin_deg:
                severity = "critical"
                would_clamp = True
                reasons.append("Joint target is too close to a limit")
                recommendations.extend(["Clamp to soft joint limit", "Reduce target range"])
            elif nearest_margin <= self.config.warn_margin_deg:
                severity = "warn"
                reasons.append("Joint target is approaching a limit")
                recommendations.extend(["Monitor joint-limit margin", "Reduce target range"])

        per_joint_margins = {}
        for i, name in enumerate(joint_names):
            if i < len(nearest_margins):
                per_joint_margins[name] = float(nearest_margins[i])

        return {
            "source": "joint_limit",
            "source_label": "Joint-limit guard",
            "profile": self.config.profile_name,
            "severity": severity,
            "reasons": sorted(set(reasons)),
            "recommendations": sorted(set(recommendations)),
            "joint_name": joint_name,
            "joint_index": idx if nearest_margins.size else None,
            "target_joint_deg": target_joint_deg,
            "lower_limit_deg": lower_limit_deg,
            "upper_limit_deg": upper_limit_deg,
            "lower_margin_deg": lower_margin_deg,
            "upper_margin_deg": upper_margin_deg,
            "margin_to_limit_deg": nearest_margin,
            "would_clamp": would_clamp,
            "soft_margin_deg": float(self.config.soft_margin_deg),
            "per_joint_margins_deg": per_joint_margins,
        }
