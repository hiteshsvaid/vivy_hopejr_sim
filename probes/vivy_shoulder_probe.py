#!/usr/bin/env python3

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np


DEFAULT_LEROBOT_REPO = Path('/home/viaan/huggingface/lerobot')
DEFAULT_SIM_CONTROLLER_PATH = Path('/home/viaan/vivy_hopejr_sim/controllers/vivy_sim_ik_controller.py')
DEFAULT_OUTPUT_PATH = Path('/tmp/vivy_shoulder_probe.json')
DEFAULT_DELTA_DEG = 5.0
DEFAULT_SETTLE_S = 0.5

_ACTIVE_PROBE = None


def _load_sim_controller_module(module_path: Path):
    spec = importlib.util.spec_from_file_location('vivy_sim_ik_controller', module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Failed to load sim controller module from {module_path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class HopeJrShoulderProbeRunner:
    BODY_PATHS = {
        'shoulder_pitch_body': '/World/JointTest/ShoulderPitchChildBody',
        'upper_arm_body': '/World/JointTest/UpperArmBody',
        'end_effector': '/World/JointTest/PalmBody/EndEffector',
    }
    JOINT_NAMES = ('right_shoulder_pitch', 'right_shoulder_yaw')

    def __init__(self, *, lerobot_repo: Path, sim_controller_path: Path, output_path: Path, delta_deg: float, settle_s: float, interval_s: float):
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
            raise RuntimeError('Isaac stage is not available; open joint_test.usda in Isaac Sim first.')
        self.baseline_model_deg = np.array([
            self.sim_module.DEFAULT_STOP_TARGETS_DEG.get(joint_name, 0.0)
            for joint_name in self.controller.model.joint_names
        ], dtype=float)
        self.results = []
        self._subscription = None
        self._last_tick_time = 0.0
        self._settle_until = 0.0
        self._phase = 'init'
        self._joint_index = -1
        self._baseline_sample = None
        self._current_probe_model_deg = None

    def _read_world_pose(self, prim_path: str):
        try:
            from pxr import UsdGeom
        except ImportError:
            raise RuntimeError('pxr.UsdGeom is not available inside Isaac Sim.')
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f'Prim not found: {prim_path}')
        xform_cache = UsdGeom.XformCache()
        world_transform = xform_cache.GetLocalToWorldTransform(prim)
        return np.array(world_transform, dtype=float).T

    def _sample_stage(self):
        sample = {}
        for name, path in self.BODY_PATHS.items():
            pose = self._read_world_pose(path)
            sample[name] = {
                'pose': pose,
                'position': pose[:3, 3].copy(),
                'rotation': pose[:3, :3].copy(),
            }
        return sample

    def _set_stage_to_model_joint_positions(self, model_joint_positions_deg: np.ndarray) -> None:
        articulation = self.controller._get_articulation()
        if articulation is None:
            raise RuntimeError('Isaac articulation is not available for /World/JointTest.')
        stage_joint_positions_deg = self.controller._model_to_stage_joint_positions_deg(model_joint_positions_deg)
        self.controller._write_joint_targets_deg(self.stage, stage_joint_positions_deg)
        self.controller._write_joint_state_deg(self.stage, stage_joint_positions_deg)
        self._settle_until = time.monotonic() + self.settle_s

    def _delta_entry(self, baseline_sample, current_sample, name):
        return {
            'baseline_position': baseline_sample[name]['position'].tolist(),
            'current_position': current_sample[name]['position'].tolist(),
            'delta_position': (current_sample[name]['position'] - baseline_sample[name]['position']).tolist(),
            'baseline_rotation': baseline_sample[name]['rotation'].tolist(),
            'current_rotation': current_sample[name]['rotation'].tolist(),
        }

    def _write_output(self):
        payload = {
            'generated_at': time.time(),
            'delta_deg': self.delta_deg,
            'settle_s': self.settle_s,
            'joint_names': list(self.JOINT_NAMES),
            'results': self.results,
        }
        tmp_path = self.output_path.with_suffix('.tmp')
        tmp_path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
        tmp_path.replace(self.output_path)
        print(f'Vivy shoulder probe: wrote {self.output_path}')

    def _record_current_joint(self):
        joint_name = self.JOINT_NAMES[self._joint_index]
        current_sample = self._sample_stage()
        result = {
            'joint_name': joint_name,
            'delta_deg': self.delta_deg,
            'probe_model_joint_positions_deg': self._current_probe_model_deg.tolist(),
            'shoulder_pitch_body': self._delta_entry(self._baseline_sample, current_sample, 'shoulder_pitch_body'),
            'upper_arm_body': self._delta_entry(self._baseline_sample, current_sample, 'upper_arm_body'),
            'end_effector': self._delta_entry(self._baseline_sample, current_sample, 'end_effector'),
        }
        self.results.append(result)
        print(
            f"Vivy shoulder probe: {joint_name} upper_arm_delta="
            f"{np.array2string(np.asarray(result['upper_arm_body']['delta_position']), precision=4, suppress_small=True)} "
            f"ee_delta={np.array2string(np.asarray(result['end_effector']['delta_position']), precision=4, suppress_small=True)}"
        )

    def _advance(self):
        if self._phase == 'init':
            articulation = self.controller._get_articulation()
            if articulation is None:
                return
            print('Vivy shoulder probe: resetting to neutral pose')
            self._set_stage_to_model_joint_positions(self.baseline_model_deg)
            self._phase = 'capture_baseline'
            return

        if time.monotonic() < self._settle_until:
            return

        if self._phase == 'capture_baseline':
            self._baseline_sample = self._sample_stage()
            self._joint_index = 0
            self._phase = 'apply_joint'
            return

        if self._phase == 'apply_joint':
            if self._joint_index >= len(self.JOINT_NAMES):
                self._write_output()
                self.stop()
                return
            joint_name = self.JOINT_NAMES[self._joint_index]
            joint_model_index = self.controller.model.joint_names.index(joint_name)
            self._current_probe_model_deg = self.baseline_model_deg.copy()
            self._current_probe_model_deg[joint_model_index] += self.delta_deg
            print(f'Vivy shoulder probe: applying {joint_name} +{self.delta_deg:.1f} deg')
            self._set_stage_to_model_joint_positions(self._current_probe_model_deg)
            self._phase = 'sample_joint'
            return

        if self._phase == 'sample_joint':
            self._record_current_joint()
            self._set_stage_to_model_joint_positions(self.baseline_model_deg)
            self._phase = 'reset_between'
            return

        if self._phase == 'reset_between':
            self._joint_index += 1
            self._phase = 'apply_joint'
            return

    def _on_update(self, _event: object) -> None:
        now = time.monotonic()
        if now - self._last_tick_time < self.interval_s:
            return
        self._last_tick_time = now
        try:
            self._advance()
        except Exception as exc:
            print(f'Vivy shoulder probe error: {exc}')
            self.stop()

    def start(self):
        import omni.kit.app
        app = omni.kit.app.get_app()
        self._subscription = app.get_update_event_stream().create_subscription_to_pop(self._on_update, name='HopeJrShoulderProbe')
        print(f'Vivy shoulder probe subscribed to Isaac update stream at {self.interval_s:.3f}s interval')
        return self

    def stop(self) -> None:
        self._subscription = None
        print('Vivy shoulder probe unsubscribed from Isaac update stream')


def start_shoulder_probe(*, lerobot_repo: str | Path = DEFAULT_LEROBOT_REPO, sim_controller_path: str | Path = DEFAULT_SIM_CONTROLLER_PATH, output_path: str | Path = DEFAULT_OUTPUT_PATH, delta_deg: float = DEFAULT_DELTA_DEG, settle_s: float = DEFAULT_SETTLE_S, interval_s: float = 0.05):
    global _ACTIVE_PROBE
    stop_shoulder_probe()
    _ACTIVE_PROBE = HopeJrShoulderProbeRunner(
        lerobot_repo=Path(lerobot_repo),
        sim_controller_path=Path(sim_controller_path),
        output_path=Path(output_path),
        delta_deg=delta_deg,
        settle_s=settle_s,
        interval_s=interval_s,
    ).start()
    return _ACTIVE_PROBE


def stop_shoulder_probe() -> None:
    global _ACTIVE_PROBE
    if _ACTIVE_PROBE is not None:
        _ACTIVE_PROBE.stop()
        _ACTIVE_PROBE = None


if __name__ == '__main__':
    start_shoulder_probe()
