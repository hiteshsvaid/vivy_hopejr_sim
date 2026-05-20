from __future__ import annotations

import time
import json
import socket
from typing import Callable, Iterable, Mapping, Sequence


LimitSpec = Mapping[str, float] | Sequence[float]
DEFAULT_CALIBRATION_LIMIT_FEEDBACK_HOST = "127.0.0.1"
DEFAULT_CALIBRATION_LIMIT_FEEDBACK_UDP_PORT = 8778


def _coerce_limit(limit: LimitSpec) -> tuple[float, float]:
    if isinstance(limit, Mapping):
        return float(limit["min"]), float(limit["max"])
    if len(limit) != 2:
        raise ValueError(f"limit sequence must contain exactly two values, got {limit!r}")
    return float(limit[0]), float(limit[1])


def detect_limit_hits(
    *,
    joint_names: Sequence[str],
    requested_deg: Sequence[float],
    limits_by_joint: Mapping[str, LimitSpec],
    tolerance_deg: float = 1e-6,
) -> list[dict]:
    if len(joint_names) != len(requested_deg):
        raise ValueError(
            f"joint_names and requested_deg length mismatch: {len(joint_names)} != {len(requested_deg)}"
        )

    hits: list[dict] = []
    tolerance = float(tolerance_deg)
    for joint_name, requested in zip(joint_names, requested_deg, strict=True):
        if joint_name not in limits_by_joint:
            continue
        min_deg, max_deg = _coerce_limit(limits_by_joint[joint_name])
        requested_value = float(requested)
        if requested_value <= min_deg + tolerance:
            hits.append(
                {
                    "joint_name": joint_name,
                    "direction": "min",
                    "requested_deg": requested_value,
                    "limit_deg": min_deg,
                    "min_deg": min_deg,
                    "max_deg": max_deg,
                    "excess_deg": max(0.0, min_deg - requested_value),
                }
            )
        elif requested_value >= max_deg - tolerance:
            hits.append(
                {
                    "joint_name": joint_name,
                    "direction": "max",
                    "requested_deg": requested_value,
                    "limit_deg": max_deg,
                    "min_deg": min_deg,
                    "max_deg": max_deg,
                    "excess_deg": max(0.0, requested_value - max_deg),
                }
            )
    return hits


def build_limit_feedback_payload(
    *,
    hits: Iterable[dict],
    sequence: int,
    limits_by_joint: Mapping[str, LimitSpec] | None = None,
    limit_source: str = "unknown",
    timestamp_ns: int | None = None,
) -> dict:
    timestamp_ns = time.time_ns() if timestamp_ns is None else int(timestamp_ns)
    joint_limits = []
    for joint_name, limit in (limits_by_joint or {}).items():
        min_deg, max_deg = _coerce_limit(limit)
        joint_limits.append({"joint_name": str(joint_name), "min_deg": min_deg, "max_deg": max_deg})
    return {
        "type": "vivy_calibration_limit_feedback",
        "source_mode": "calibration_preview",
        "timestamp": timestamp_ns / 1_000_000_000.0,
        "timestamp_ns": timestamp_ns,
        "sequence": int(sequence),
        "limit_hits": [dict(hit) for hit in hits],
        "joint_limits": joint_limits,
        "limit_source": str(limit_source),
    }


class CalibrationLimitFeedbackPublisher:
    def __init__(
        self,
        *,
        host: str = DEFAULT_CALIBRATION_LIMIT_FEEDBACK_HOST,
        port: int = DEFAULT_CALIBRATION_LIMIT_FEEDBACK_UDP_PORT,
        socket_factory: Callable[[], socket.socket] | None = None,
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self._socket = (
            socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            if socket_factory is None
            else socket_factory()
        )
        self._sequence = 0

    def publish_hits(self, hits: Iterable[dict]) -> dict | None:
        return self.publish_status(hits=hits)

    def publish_status(
        self,
        *,
        hits: Iterable[dict],
        limits_by_joint: Mapping[str, LimitSpec] | None = None,
        limit_source: str = "unknown",
    ) -> dict | None:
        hit_list = [dict(hit) for hit in hits]
        limit_map = dict(limits_by_joint or {})
        if not hit_list and not limit_map:
            return None
        self._sequence += 1
        payload = build_limit_feedback_payload(
            hits=hit_list,
            limits_by_joint=limit_map,
            limit_source=limit_source,
            sequence=self._sequence,
        )
        self._socket.sendto(json.dumps(payload).encode("utf-8"), (self.host, self.port))
        return payload
