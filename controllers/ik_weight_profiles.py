#!/usr/bin/env python3

from __future__ import annotations

from typing import Iterable

STAGE_WEIGHT_PROFILES: dict[str, dict[str, float]] = {
    "baseline_v1": {
        "right_shoulder_pitch": 1.0,
        "right_shoulder_yaw": 1.0,
        "right_arm_twist": 1.0,
        "right_elbow": 1.0,
        "right_forearm_twist": 0.7,
        "right_wrist": 0.45,
        "right_palm": 0.35,
    },
    "shoulder_elbow_heavy_v1": {
        "right_shoulder_pitch": 1.2,
        "right_shoulder_yaw": 1.2,
        "right_arm_twist": 0.3,
        "right_elbow": 1.2,
        "right_forearm_twist": 0.0,
        "right_wrist": 0.4,
        "right_palm": 0.2,
    },
}

DEFAULT_STAGE_WEIGHT_PROFILE = "upper_elbow_light_v1"


def list_stage_weight_profiles() -> list[str]:
    return sorted(STAGE_WEIGHT_PROFILES.keys())


def build_stage_joint_weights(joint_names: Iterable[str], profile_name: str) -> list[float]:
    try:
        profile = STAGE_WEIGHT_PROFILES[profile_name]
    except KeyError as exc:
        available = ", ".join(list_stage_weight_profiles())
        raise ValueError(f"Unknown stage weight profile: {profile_name}. Available: {available}") from exc
    return [float(profile.get(name, 1.0)) for name in joint_names]
