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
    count = 0
    sum_abs = [0.0, 0.0, 0.0]
    sum_norm = 0.0
    for row in rows:
        if row.get("status") != "applied":
            continue
        result = row.get("result") or {}
        err = result.get("stage_end_effector_error")
        if err is None:
            continue
        count += 1
        for i, value in enumerate(err):
            sum_abs[i] += abs(float(value))
        sum_norm += _error_norm(err)

    if count == 0:
        print("no applied rows with stage_end_effector_error found")
        return

    mean_abs = [v / count for v in sum_abs]
    mean_norm = float(sum_norm) / count
    print(f"samples={count}")
    print("  mean_abs_err_m=", [round(v, 5) for v in mean_abs])
    print("  mean_err_norm_m=", round(mean_norm, 5))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Hope Jr IK event log error")
    parser.add_argument("--path", type=Path, default=Path("/tmp/hope_jr_sim_ik_events.ndjson"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summarize(_load_rows(args.path))


if __name__ == "__main__":
    main()
