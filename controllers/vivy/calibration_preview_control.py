from __future__ import annotations


def is_calibration_preview_payload(payload: dict) -> bool:
    return bool(payload.get("calibration_preview", False)) or payload.get("source_mode") == "calibration_preview"


def build_calibration_preview_control_state() -> dict:
    return {
        "type": "vivy_control_state",
        "source": "calibration_preview",
        "status": "preview_enabled",
        "right": {"enabled": True},
        "left": {"enabled": True},
        "head": {"enabled": True, "armed_by": "calibration_preview"},
    }
