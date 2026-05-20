#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/home/viaan/vivy_hopejr_sim")
sys.path.insert(0, str(ROOT / "controllers" / "vivy"))

from calibration_limit_feedback import (  # noqa: E402
    CalibrationLimitFeedbackPublisher,
    build_limit_feedback_payload,
    detect_limit_hits,
)


def test_inside_limits_has_no_hits() -> None:
    hits = detect_limit_hits(
        joint_names=["right_wrist"],
        requested_deg=[10.0],
        limits_by_joint={"right_wrist": {"min": -30.0, "max": 30.0}},
    )
    if hits:
        raise AssertionError(f"expected no hits, got {hits}")


def test_below_min_reports_min_hit() -> None:
    hits = detect_limit_hits(
        joint_names=["right_wrist"],
        requested_deg=[-45.0],
        limits_by_joint={"right_wrist": {"min": -30.0, "max": 30.0}},
    )
    if len(hits) != 1:
        raise AssertionError(f"expected one hit, got {hits}")
    hit = hits[0]
    if hit["joint_name"] != "right_wrist" or hit["direction"] != "min":
        raise AssertionError(f"unexpected min hit: {hit}")
    if hit["requested_deg"] != -45.0 or hit["limit_deg"] != -30.0 or hit["excess_deg"] != 15.0:
        raise AssertionError(f"unexpected min hit values: {hit}")


def test_above_max_reports_max_hit() -> None:
    hits = detect_limit_hits(
        joint_names=["right_wrist"],
        requested_deg=[42.0],
        limits_by_joint={"right_wrist": (-30.0, 30.0)},
    )
    if len(hits) != 1:
        raise AssertionError(f"expected one hit, got {hits}")
    hit = hits[0]
    if hit["joint_name"] != "right_wrist" or hit["direction"] != "max":
        raise AssertionError(f"unexpected max hit: {hit}")
    if hit["requested_deg"] != 42.0 or hit["limit_deg"] != 30.0 or hit["excess_deg"] != 12.0:
        raise AssertionError(f"unexpected max hit values: {hit}")


def test_at_limit_reports_hit() -> None:
    hits = detect_limit_hits(
        joint_names=["right_wrist", "right_palm"],
        requested_deg=[-30.0, 40.0],
        limits_by_joint={
            "right_wrist": {"min": -30.0, "max": 30.0},
            "right_palm": {"min": -20.0, "max": 40.0},
        },
    )
    got = [(hit["joint_name"], hit["direction"], hit["excess_deg"]) for hit in hits]
    expected = [("right_wrist", "min", 0.0), ("right_palm", "max", 0.0)]
    if got != expected:
        raise AssertionError(f"unexpected at-limit hits: {got}")


def test_multiple_joints_reports_only_hits() -> None:
    hits = detect_limit_hits(
        joint_names=["right_wrist", "right_palm", "right_thumb"],
        requested_deg=[0.0, 50.0, -90.0],
        limits_by_joint={
            "right_wrist": (-30.0, 30.0),
            "right_palm": (-20.0, 40.0),
            "right_thumb": (-60.0, 20.0),
        },
    )
    got = [(hit["joint_name"], hit["direction"]) for hit in hits]
    expected = [("right_palm", "max"), ("right_thumb", "min")]
    if got != expected:
        raise AssertionError(f"unexpected hits: {got}")


def test_build_limit_feedback_payload() -> None:
    hits = [{"joint_name": "right_wrist", "direction": "max", "requested_deg": 42.0}]
    payload = build_limit_feedback_payload(
        hits=hits,
        limits_by_joint={"right_wrist": {"min": -30.0, "max": 30.0}},
        limit_source="isaac_stage",
        sequence=3,
        timestamp_ns=123_000_000_000,
    )
    if payload["type"] != "vivy_calibration_limit_feedback":
        raise AssertionError(f"unexpected payload type: {payload['type']}")
    if payload["source_mode"] != "calibration_preview":
        raise AssertionError(f"unexpected source mode: {payload['source_mode']}")
    if payload["limit_source"] != "isaac_stage":
        raise AssertionError(f"unexpected limit source: {payload['limit_source']}")
    if payload["sequence"] != 3 or payload["timestamp"] != 123.0:
        raise AssertionError(f"unexpected payload timing: {payload}")
    if payload["limit_hits"] != hits:
        raise AssertionError(f"unexpected hits payload: {payload['limit_hits']}")
    if payload["joint_limits"] != [{"joint_name": "right_wrist", "min_deg": -30.0, "max_deg": 30.0}]:
        raise AssertionError(f"unexpected joint limits payload: {payload['joint_limits']}")


class FakeSocket:
    def __init__(self) -> None:
        self.sent: list[tuple[bytes, tuple[str, int]]] = []

    def sendto(self, payload: bytes, address: tuple[str, int]) -> None:
        self.sent.append((payload, address))


def test_publisher_skips_empty_hits_and_sends_limit_hits() -> None:
    fake_socket = FakeSocket()
    publisher = CalibrationLimitFeedbackPublisher(
        host="127.0.0.1",
        port=8778,
        socket_factory=lambda: fake_socket,
    )

    empty_payload = publisher.publish_hits([])
    if empty_payload is not None:
        raise AssertionError(f"empty hits should not publish, got {empty_payload}")
    if fake_socket.sent:
        raise AssertionError("empty hits should not send UDP payload")

    hit = {"joint_name": "right_wrist", "direction": "max", "requested_deg": 42.0}
    payload = publisher.publish_hits([hit])
    if payload is None:
        raise AssertionError("expected non-empty hits to publish")
    if payload["sequence"] != 1:
        raise AssertionError(f"unexpected publisher sequence: {payload['sequence']}")
    if payload["limit_hits"] != [hit]:
        raise AssertionError(f"unexpected publisher hits: {payload['limit_hits']}")
    if len(fake_socket.sent) != 1 or fake_socket.sent[0][1] != ("127.0.0.1", 8778):
        raise AssertionError(f"unexpected sent UDP payloads: {fake_socket.sent}")


def test_publisher_sends_limits_without_hits() -> None:
    fake_socket = FakeSocket()
    publisher = CalibrationLimitFeedbackPublisher(
        host="127.0.0.1",
        port=8778,
        socket_factory=lambda: fake_socket,
    )

    payload = publisher.publish_status(
        hits=[],
        limits_by_joint={"right_wrist": {"min": -30.0, "max": 30.0}},
        limit_source="isaac_stage",
    )
    if payload is None:
        raise AssertionError("expected live limits to publish even without hits")
    if payload["limit_hits"] != []:
        raise AssertionError(f"unexpected hits payload: {payload['limit_hits']}")
    if payload["joint_limits"] != [{"joint_name": "right_wrist", "min_deg": -30.0, "max_deg": 30.0}]:
        raise AssertionError(f"unexpected joint limits: {payload['joint_limits']}")
    if payload["limit_source"] != "isaac_stage":
        raise AssertionError(f"unexpected limit source: {payload['limit_source']}")
    if len(fake_socket.sent) != 1:
        raise AssertionError(f"expected one UDP send, got {fake_socket.sent}")


def main() -> int:
    test_inside_limits_has_no_hits()
    test_below_min_reports_min_hit()
    test_above_max_reports_max_hit()
    test_at_limit_reports_hit()
    test_multiple_joints_reports_only_hits()
    test_build_limit_feedback_payload()
    test_publisher_skips_empty_hits_and_sends_limit_hits()
    test_publisher_sends_limits_without_hits()
    print("[vivy-smoke] Vivy calibration limit feedback tests ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
