from __future__ import annotations


def is_calibration_preview_payload(payload: dict) -> bool:
    return bool(payload.get("calibration_preview", False)) or payload.get("source_mode") == "calibration_preview"


def extract_calibration_limit_updates(payload: dict) -> dict[str, tuple[float, float]]:
    updates: dict[str, tuple[float, float]] = {}
    for item in payload.get("calibration_limit_updates") or []:
        if not isinstance(item, dict):
            continue
        joint_name = item.get("joint_name")
        if not isinstance(joint_name, str):
            continue
        try:
            min_deg = float(item["min_deg"])
            max_deg = float(item["max_deg"])
        except (KeyError, TypeError, ValueError):
            continue
        if min_deg <= max_deg:
            updates[joint_name] = (min_deg, max_deg)
    return updates


def is_calibration_limit_request(payload: dict) -> bool:
    return is_calibration_preview_payload(payload) and bool(payload.get("calibration_limit_request", False))


def build_calibration_preview_control_state() -> dict:
    return {
        "type": "vivy_control_state",
        "source": "calibration_preview",
        "status": "preview_enabled",
        "right": {"enabled": True},
        "left": {"enabled": True},
        "head": {"enabled": True, "armed_by": "calibration_preview"},
    }
