from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TeleopSafetyAdvisorConfig:
    profile_name: str
    forward_mapped_delta_warn_m: float
    forward_mapped_delta_critical_m: float
    end_effector_error_warn_m: float
    end_effector_error_critical_m: float
    joint_step_warn_deg: float
    joint_step_critical_deg: float


ADVISOR_PROFILES: dict[str, TeleopSafetyAdvisorConfig] = {
    "teleop_advisory_v1": TeleopSafetyAdvisorConfig(
        profile_name="teleop_advisory_v1",
        forward_mapped_delta_warn_m=0.10,
        forward_mapped_delta_critical_m=0.20,
        end_effector_error_warn_m=0.08,
        end_effector_error_critical_m=0.10,
        joint_step_warn_deg=6.0,
        joint_step_critical_deg=8.0,
    ),
}

DEFAULT_TELEOP_ADVISOR_PROFILE = "teleop_advisory_v1"


class TeleopSafetyAdvisor:
    def __init__(self, profile_name: str = DEFAULT_TELEOP_ADVISOR_PROFILE) -> None:
        if profile_name not in ADVISOR_PROFILES:
            raise ValueError(
                f"Unknown teleop safety advisor profile '{profile_name}'. Available: {sorted(ADVISOR_PROFILES)}"
            )
        self.config = ADVISOR_PROFILES[profile_name]

    def evaluate(
        self,
        *,
        mapped_delta: np.ndarray | list[float] | tuple[float, ...] | None,
        stage_end_effector_error: np.ndarray | list[float] | tuple[float, ...] | None,
        stage_dls_delta_deg: np.ndarray | list[float] | tuple[float, ...] | None,
        joint_names: list[str],
    ) -> dict[str, Any]:
        mapped = np.asarray(mapped_delta, dtype=float) if mapped_delta is not None else None
        ee_error = np.asarray(stage_end_effector_error, dtype=float) if stage_end_effector_error is not None else None
        joint_step = np.asarray(stage_dls_delta_deg, dtype=float) if stage_dls_delta_deg is not None else None

        recommendations: list[str] = []
        reasons: list[str] = []
        severity = "ok"

        forward_mapped_delta_abs = float(abs(mapped[0])) if mapped is not None and mapped.size >= 1 else None
        end_effector_error_norm_m = float(np.linalg.norm(ee_error)) if ee_error is not None else None

        joint_step_abs_max_deg = None
        joint_step_abs_max_joint = None
        if joint_step is not None and joint_step.size:
            step_index = int(np.argmax(np.abs(joint_step)))
            joint_step_abs_max_deg = float(abs(joint_step[step_index]))
            joint_step_abs_max_joint = joint_names[step_index] if step_index < len(joint_names) else None

        if forward_mapped_delta_abs is not None:
            if forward_mapped_delta_abs >= self.config.forward_mapped_delta_critical_m:
                severity = "critical"
                reasons.append("Forward reach is near the tested limit")
                recommendations.extend(["Clamp forward target", "Reduce forward test range"])
            elif forward_mapped_delta_abs >= self.config.forward_mapped_delta_warn_m:
                severity = "warn" if severity == "ok" else severity
                reasons.append("Forward reach is approaching the tested limit")
                recommendations.append("Monitor forward reach")

        if end_effector_error_norm_m is not None:
            if end_effector_error_norm_m >= self.config.end_effector_error_critical_m:
                severity = "critical"
                reasons.append("Tracking error is too large")
                recommendations.extend(["Freeze or re-anchor if this grows", "Clamp target step"])
            elif end_effector_error_norm_m >= self.config.end_effector_error_warn_m:
                severity = "warn" if severity == "ok" else severity
                reasons.append("Tracking error is growing")
                recommendations.append("Clamp target step")

        if joint_step_abs_max_deg is not None:
            if joint_step_abs_max_deg >= self.config.joint_step_critical_deg:
                severity = "critical"
                reasons.append("Joint motion jumped too much in one step")
                recommendations.extend(["Limit joint step", "Freeze or re-anchor if this grows"])
            elif joint_step_abs_max_deg >= self.config.joint_step_warn_deg:
                severity = "warn" if severity == "ok" else severity
                reasons.append("Joint step is getting large")
                recommendations.append("Limit joint step")

        recommendations = sorted(set(recommendations))
        reasons = sorted(set(reasons))
        return {
            "profile": self.config.profile_name,
            "severity": severity,
            "reasons": reasons,
            "recommendations": recommendations,
            "forward_mapped_delta_abs_m": forward_mapped_delta_abs,
            "end_effector_error_norm_m": end_effector_error_norm_m,
            "joint_step_abs_max_deg": joint_step_abs_max_deg,
            "joint_step_abs_max_joint": joint_step_abs_max_joint,
        }
