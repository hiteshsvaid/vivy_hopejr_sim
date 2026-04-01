#!/usr/bin/env python3

from __future__ import annotations

from typing import Iterable

JOINT_CONTROL_SOLVE = "solve"
JOINT_CONTROL_HOLD_START = "hold_start"
JOINT_CONTROL_HOLD_LAST = "hold_last"

JOINT_CONTROL_PROFILES: dict[str, dict[str, str]] = {
    "all_solve_v1": {},
    "position_only_hold_distal_v1": {
        "right_arm_twist": JOINT_CONTROL_HOLD_START,
        "right_forearm_twist": JOINT_CONTROL_HOLD_START,
        "right_wrist": JOINT_CONTROL_HOLD_START,
        "right_palm": JOINT_CONTROL_HOLD_START,
    },
}

DEFAULT_POSITION_ONLY_JOINT_CONTROL_PROFILE = "position_only_hold_distal_v1"


def list_joint_control_profiles() -> list[str]:
    return sorted(JOINT_CONTROL_PROFILES.keys())


def build_joint_control_modes(joint_names: Iterable[str], profile_name: str) -> list[str]:
    try:
        profile = JOINT_CONTROL_PROFILES[profile_name]
    except KeyError as exc:
        available = ", ".join(list_joint_control_profiles())
        raise ValueError(f"Unknown joint control profile: {profile_name}. Available: {available}") from exc
    return [str(profile.get(name, JOINT_CONTROL_SOLVE)) for name in joint_names]
