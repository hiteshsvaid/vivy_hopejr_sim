#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/home/viaan/vivy_hopejr_sim")
sys.path.insert(0, str(ROOT / "controllers" / "vivy"))

from calibration_preview_control import (  # noqa: E402
    build_calibration_preview_control_state,
    is_calibration_preview_payload,
)


def _apply_control_gate_like_viewer(payload: dict, panel_payload: dict, control_state: dict | None) -> None:
    if is_calibration_preview_payload(payload):
        preview_control_state = build_calibration_preview_control_state()
        payload["control_state"] = preview_control_state
        payload["control_state_source"] = "calibration_preview"
        panel_payload["control_state"] = dict(preview_control_state)
        panel_payload["control_state_source"] = "calibration_preview"
        return
    if control_state is None:
        payload["control_state"] = {
            "type": "vivy_control_state",
            "source": "udp",
            "status": "waiting_for_udp",
            "right": {"enabled": False},
            "left": {"enabled": False},
            "head": {"enabled": False, "armed_by": None},
        }
        payload["control_state_source"] = "udp_waiting"
        control_state = payload["control_state"]
    for side in ("right", "left"):
        payload[f"{side}_follow_target_enabled"] = bool(control_state.get(side, {}).get("enabled", False))
        if side == "right":
            payload["follow_target_enabled"] = payload[f"{side}_follow_target_enabled"]


def test_calibration_preview_bypasses_missing_control_udp() -> None:
    payload = {
        "source_mode": "calibration_preview",
        "calibration_preview": True,
        "follow_target_enabled": True,
        "waiting_for_anchor": False,
    }
    panel_payload = {}

    _apply_control_gate_like_viewer(payload, panel_payload, control_state=None)

    if payload.get("control_state_source") != "calibration_preview":
        raise AssertionError(f"unexpected control source: {payload.get('control_state_source')}")
    control_state = payload.get("control_state")
    if not isinstance(control_state, dict):
        raise AssertionError("missing calibration preview control state")
    for target in ("right", "left", "head"):
        if not bool(control_state.get(target, {}).get("enabled", False)):
            raise AssertionError(f"{target} was not enabled for calibration preview")


def test_normal_payload_waits_for_control_udp() -> None:
    payload = {"source_mode": "live"}
    panel_payload = {}

    _apply_control_gate_like_viewer(payload, panel_payload, control_state=None)

    if payload.get("control_state_source") != "udp_waiting":
        raise AssertionError(f"unexpected control source: {payload.get('control_state_source')}")
    if bool(payload.get("follow_target_enabled", True)):
        raise AssertionError("normal payload should not enable follow target without control UDP")


def main() -> int:
    test_calibration_preview_bypasses_missing_control_udp()
    test_normal_payload_waits_for_control_udp()
    print("[vivy-smoke] Vivy target viewer calibration preview gate tests ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
