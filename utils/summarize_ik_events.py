#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _error_norm(err: list[float]) -> float:
    return math.sqrt(sum(float(v) * float(v) for v in err))


def summarize(rows: list[dict]) -> None:
    buckets: dict[str, dict[str, object]] = {}
    for row in rows:
        if row.get("status") != "applied":
            continue
        result = row.get("result") or {}
        err = result.get("stage_end_effector_error")
        profile = result.get("stage_weight_profile") or "unknown"
        if err is None:
            continue
        bucket = buckets.setdefault(profile, {"count": 0, "sum_abs": [0.0, 0.0, 0.0], "sum_norm": 0.0})
        bucket["count"] = int(bucket["count"]) + 1
        for i, value in enumerate(err):
            bucket["sum_abs"][i] += abs(float(value))
        bucket["sum_norm"] += _error_norm(err)

    if not buckets:
        print("no applied rows with stage_end_effector_error found")
        return

    for profile, bucket in sorted(buckets.items()):
        count = int(bucket["count"])
        mean_abs = [v / count for v in bucket["sum_abs"]]
        mean_norm = float(bucket["sum_norm"]) / count
        print(f"profile={profile} samples={count}")
        print("  mean_abs_err_m=", [round(v, 5) for v in mean_abs])
        print("  mean_err_norm_m=", round(mean_norm, 5))


def _default_profile_event_path(profile: str) -> Path:
    return Path(f"/tmp/hope_jr_sim_ik_events_{profile}.ndjson")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Hope Jr IK event log error by weight profile")
    parser.add_argument("--path", type=Path, default=None)
    parser.add_argument("--profile", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = args.path
    if path is None:
        if args.profile is None:
            path = Path("/tmp/hope_jr_sim_ik_events.ndjson")
        else:
            path = _default_profile_event_path(args.profile)
    summarize(_load_rows(path))


if __name__ == "__main__":
    main()
