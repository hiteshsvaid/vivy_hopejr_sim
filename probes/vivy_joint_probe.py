#!/usr/bin/env python3

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_LEROBOT_REPO = Path("/home/viaan/huggingface/lerobot")
DEFAULT_SIM_CONTROLLER_PATH = Path("/home/viaan/vivy_hopejr_sim/controllers/vivy_sim_ik_controller.py")
DEFAULT_OUTPUT_PATH = Path("/tmp/vivy_joint_probe.json")
DEFAULT_DELTA_DEG = 5.0
DEFAULT_SETTLE_S = 0.5

_ACTIVE_PROBE = None


def _load_sim_controller_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("vivy_sim_ik_controller", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load sim controller module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class HopeJrJointProbeRunner:
    def __init__(
        self,
        *,
        lerobot_repo: Path,
        sim_controller_path: Path,
        output_path: Path,
        delta_deg: float,
        settle_s: float,
        interval_s: float,
    ):
        self.output_path = output_path
        self.delta_deg = float(delta_deg)
        self.settle_s = float(settle_s)
        self.interval_s = float(interval_s)
        self.sim_module = _load_sim_controller_module(sim_controller_path)
        self.controller = self.sim_module.VivySimIkController(
            lerobot_repo=lerobot_repo,
            packet_path=self.sim_module.DEFAULT_PACKET_PATH,
            joint_root_path=self.sim_module.DEFAULT_JOINT_ROOT_PATH,
            position_scale=1.0,
            world_offset=np.zeros(3, dtype=float),
            world_rotate_xyz_deg=np.zeros(3, dtype=float),
            quest_position_axes=(0, 1, 2),
            quest_position_signs=np.ones(3, dtype=float),
            position_only=True,
            debug_path=self.sim_module.DEFAULT_DEBUG_PATH,
            use_udp=False,
            udp_listen_host=self.sim_module.DEFAULT_UDP_LISTEN_HOST,
            udp_listen_port=self.sim_module.DEFAULT_UDP_LISTEN_PORT,
            teleop_debug_root=self.sim_module.DEFAULT_TELEOP_DEBUG_ROOT,
            show_teleop_debug=False,
            anchor_delay_s=0.0,
            grip_threshold=0.0,
            event_log_path=self.sim_module.DEFAULT_EVENT_LOG_PATH,
            quest_deadband_m=0.0,
            packet_stale_timeout_s=self.sim_module.DEFAULT_PACKET_STALE_TIMEOUT_S,
            end_effector_path=self.sim_module.DEFAULT_END_EFFECTOR_PATH,
            write_joint_state_directly=True,
        )
        self.stage = self.controller._get_stage()
        if self.stage is None:
            raise RuntimeError("Isaac stage is not available; open joint_test.usda in Isaac Sim first.")
        self.articulation = None
        self.baseline_model_deg = np.array(
            [
                self.sim_module.DEFAULT_STOP_TARGETS_DEG.get(joint_name, 0.0)
                for joint_name in self.controller.model.joint_names
            ],
            dtype=float,
        )
        self.results: list[dict[str, Any]] = []
        self._subscription = None
        self._last_tick_time = 0.0
        self._settle_until = 0.0
        self._phase = "init"
        self._joint_index = -1
        self._baseline_sample = None
        self._current_probe_model_deg = None

    def _set_stage_to_model_joint_positions(self, model_joint_positions_deg: np.ndarray) -> None:
        stage_joint_positions_deg = self.controller._model_to_stage_joint_positions_deg(model_joint_positions_deg)
        self.controller._write_joint_targets_deg(self.stage, stage_joint_positions_deg)
        self.controller._write_joint_state_deg(self.stage, stage_joint_positions_deg)
        self._settle_until = time.monotonic() + self.settle_s

    def _sample_pose(self) -> dict[str, Any]:
        stage_pose = self.controller._read_stage_end_effector_pose(self.stage)
        if stage_pose is None:
            raise RuntimeError("Failed to read stage EndEffector pose.")
        stage_joint_positions_deg = self.controller._read_stage_joint_positions_deg(self.stage)
        model_joint_positions_deg = None
        if stage_joint_positions_deg is not None:
            model_joint_positions_deg = self.controller._stage_to_model_joint_positions_deg(stage_joint_positions_deg)
        return {
            "stage_end_effector_position": stage_pose[:3, 3].copy(),
            "stage_joint_positions_deg": None if stage_joint_positions_deg is None else stage_joint_positions_deg.copy(),
            "model_joint_positions_deg": None if model_joint_positions_deg is None else model_joint_positions_deg.copy(),
        }

    def _write_output(self) -> None:
        payload = {
            "generated_at": time.time(),
            "delta_deg": self.delta_deg,
            "settle_s": self.settle_s,
            "joint_names": self.controller.model.joint_names,
            "baseline": {
                "model_joint_positions_deg": self.baseline_model_deg.tolist(),
                "model_end_effector_position": self._baseline_sample["model_end_effector_position"].tolist(),
                "stage_end_effector_position": self._baseline_sample["stage_end_effector_position"].tolist(),
                "stage_joint_positions_deg": None if self._baseline_sample["stage_joint_positions_deg"] is None else self._baseline_sample["stage_joint_positions_deg"].tolist(),
                "stage_model_joint_positions_deg": None if self._baseline_sample["model_joint_positions_deg"] is None else self._baseline_sample["model_joint_positions_deg"].tolist(),
            },
            "results": self.results,
        }
        tmp_path = self.output_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(self.output_path)
        print(f"Vivy joint probe: wrote {self.output_path}")

    def _record_current_joint(self) -> None:
        joint_name = self.controller.model.joint_names[self._joint_index]
        stage_sample = self._sample_pose()
        probe_fk = self.controller.model.forward_kinematics(self._current_probe_model_deg)
        baseline_model_position = self._baseline_sample["model_end_effector_position"]
        baseline_stage_position = self._baseline_sample["stage_end_effector_position"]
        probe_stage_position = stage_sample["stage_end_effector_position"]
        probe_model_position = probe_fk[:3, 3]
        model_delta = probe_model_position - baseline_model_position
        probe_model_stage_pose = self._baseline_sample["model_to_stage_transform"] @ probe_fk
        baseline_model_stage_position = self._baseline_sample["model_to_stage_transform"][:3, 3]
        model_stage_delta = probe_model_stage_pose[:3, 3] - baseline_model_stage_position
        stage_delta = probe_stage_position - baseline_stage_position
        delta_error = stage_delta - model_stage_delta
        result = {
            "joint_name": joint_name,
            "delta_deg": self.delta_deg,
            "baseline_model_joint_positions_deg": self.baseline_model_deg.tolist(),
            "probe_model_joint_positions_deg": self._current_probe_model_deg.tolist(),
            "baseline_model_end_effector_position": baseline_model_position.tolist(),
            "probe_model_end_effector_position": probe_model_position.tolist(),
            "baseline_stage_end_effector_position": baseline_stage_position.tolist(),
            "probe_stage_end_effector_position": probe_stage_position.tolist(),
            "model_delta": model_delta.tolist(),
            "model_stage_delta": model_stage_delta.tolist(),
            "stage_delta": stage_delta.tolist(),
            "delta_error": delta_error.tolist(),
            "stage_joint_positions_deg": None if stage_sample["stage_joint_positions_deg"] is None else stage_sample["stage_joint_positions_deg"].tolist(),
            "model_joint_positions_deg": None if stage_sample["model_joint_positions_deg"] is None else stage_sample["model_joint_positions_deg"].tolist(),
        }
        self.results.append(result)
        print(
            f"Vivy joint probe: {joint_name} model_stage={np.array2string(model_stage_delta, precision=4, suppress_small=True)} "
            f"stage={np.array2string(stage_delta, precision=4, suppress_small=True)} "
            f"error={np.array2string(delta_error, precision=4, suppress_small=True)}"
        )

    def _advance(self) -> None:
        if self._phase == "init":
            self.articulation = self.controller._get_articulation()
            if self.articulation is None:
                return
            print("Vivy joint probe: resetting to neutral pose")
            self._set_stage_to_model_joint_positions(self.baseline_model_deg)
            self._phase = "capture_baseline"
            return

        if time.monotonic() < self._settle_until:
            return

        if self._phase == "capture_baseline":
            baseline_stage = self._sample_pose()
            baseline_fk = self.controller.model.forward_kinematics(self.baseline_model_deg)
            baseline_stage_pose = self.controller._read_stage_end_effector_pose(self.stage)
            if baseline_stage_pose is None:
                raise RuntimeError("Failed to read stage EndEffector pose for baseline.")
            model_to_stage_transform = baseline_stage_pose @ np.linalg.inv(baseline_fk)
            self._baseline_sample = {
                "stage_end_effector_position": baseline_stage["stage_end_effector_position"],
                "stage_joint_positions_deg": baseline_stage["stage_joint_positions_deg"],
                "model_joint_positions_deg": baseline_stage["model_joint_positions_deg"],
                "model_end_effector_position": baseline_fk[:3, 3].copy(),
                "model_to_stage_transform": model_to_stage_transform.copy(),
            }
            self._joint_index = 0
            self._phase = "apply_joint"
            return

        if self._phase == "apply_joint":
            if self._joint_index >= len(self.controller.model.joint_names):
                self._write_output()
                self.stop()
                return
            joint_name = self.controller.model.joint_names[self._joint_index]
            self._current_probe_model_deg = self.baseline_model_deg.copy()
            self._current_probe_model_deg[self._joint_index] += self.delta_deg
            print(f"Vivy joint probe: applying {joint_name} +{self.delta_deg:.1f} deg")
            self._set_stage_to_model_joint_positions(self._current_probe_model_deg)
            self._phase = "sample_joint"
            return

        if self._phase == "sample_joint":
            self._record_current_joint()
            self._set_stage_to_model_joint_positions(self.baseline_model_deg)
            self._phase = "reset_between"
            return

        if self._phase == "reset_between":
            self._joint_index += 1
            self._phase = "apply_joint"
            return

    def _on_update(self, _event: object) -> None:
        now = time.monotonic()
        if now - self._last_tick_time < self.interval_s:
            return
        self._last_tick_time = now
        try:
            self._advance()
        except Exception as exc:
            print(f"Vivy joint probe error: {exc}")
            self.stop()

    def start(self):
        import omni.kit.app

        app = omni.kit.app.get_app()
        self._subscription = app.get_update_event_stream().create_subscription_to_pop(
            self._on_update,
            name="HopeJrJointProbe",
        )
        print(
            f"Vivy joint probe subscribed to Isaac update stream at {self.interval_s:.3f}s interval"
        )
        return self

    def stop(self) -> None:
        self._subscription = None
        print("Vivy joint probe unsubscribed from Isaac update stream")


def start_joint_probe(
    *,
    lerobot_repo: str | Path = DEFAULT_LEROBOT_REPO,
    sim_controller_path: str | Path = DEFAULT_SIM_CONTROLLER_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    delta_deg: float = DEFAULT_DELTA_DEG,
    settle_s: float = DEFAULT_SETTLE_S,
    interval_s: float = 0.05,
):
    global _ACTIVE_PROBE
    stop_joint_probe()
    _ACTIVE_PROBE = HopeJrJointProbeRunner(
        lerobot_repo=Path(lerobot_repo),
        sim_controller_path=Path(sim_controller_path),
        output_path=Path(output_path),
        delta_deg=delta_deg,
        settle_s=settle_s,
        interval_s=interval_s,
    ).start()
    return _ACTIVE_PROBE


def stop_joint_probe() -> None:
    global _ACTIVE_PROBE
    if _ACTIVE_PROBE is not None:
        _ACTIVE_PROBE.stop()
        _ACTIVE_PROBE = None


if __name__ == "__main__":
    start_joint_probe()
