#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import numpy as np


class HopeJrStageIo:
    def __init__(
        self,
        *,
        articulation_root_path: str,
        joint_root_path: str,
        end_effector_path: str,
        joint_names: list[str] | tuple[str, ...],
        model_joint_signs: np.ndarray,
    ):
        self.articulation_root_path = articulation_root_path
        self.joint_root_path = joint_root_path.rstrip("/")
        self.end_effector_path = end_effector_path
        self.joint_names = tuple(joint_names)
        self.model_joint_signs = np.asarray(model_joint_signs, dtype=float)
        self._articulation = None
        self._articulation_joint_indices = None

    def get_stage(self):
        try:
            import omni.usd
        except ImportError:
            return None
        return omni.usd.get_context().get_stage()

    def read_stage_end_effector_pose(self, stage) -> np.ndarray | None:
        if stage is None:
            return None
        try:
            from pxr import UsdGeom
        except ImportError:
            return None
        prim = stage.GetPrimAtPath(self.end_effector_path)
        if not prim.IsValid():
            return None
        try:
            xform_cache = UsdGeom.XformCache()
            world_transform = xform_cache.GetLocalToWorldTransform(prim)
            matrix = np.array(world_transform, dtype=float).T
            return matrix
        except Exception:
            return None

    def read_stage_end_effector_position(self, stage) -> np.ndarray | None:
        pose = self.read_stage_end_effector_pose(stage)
        if pose is None:
            return None
        return pose[:3, 3].copy()

    def _get_articulation(self):
        try:
            from isaacsim.core.prims import SingleArticulation
        except ImportError:
            return None
        if self._articulation is None:
            self._articulation = SingleArticulation(self.articulation_root_path, reset_xform_properties=False)
            try:
                self._articulation.initialize()
            except Exception:
                self._articulation = None
                return None
        return self._articulation

    def _get_articulation_joint_indices(self) -> np.ndarray | None:
        articulation = self._get_articulation()
        if articulation is None:
            return None
        if self._articulation_joint_indices is None:
            try:
                self._articulation_joint_indices = np.array(
                    [articulation.get_dof_index(name) for name in self.joint_names],
                    dtype=np.int64,
                )
            except Exception:
                return None
        return self._articulation_joint_indices

    def stage_to_model_joint_positions_deg(self, joint_positions_deg: np.ndarray) -> np.ndarray:
        return np.asarray(joint_positions_deg, dtype=float) * self.model_joint_signs

    def model_to_stage_joint_positions_deg(self, joint_positions_deg: np.ndarray) -> np.ndarray:
        return np.asarray(joint_positions_deg, dtype=float) * self.model_joint_signs

    def read_current_joint_targets_deg(self, stage) -> np.ndarray:
        articulation = self._get_articulation()
        joint_indices = self._get_articulation_joint_indices()
        if articulation is not None and joint_indices is not None:
            try:
                joint_positions_rad = articulation.get_joint_positions(joint_indices=joint_indices)
                if joint_positions_rad is not None:
                    return np.rad2deg(np.asarray(joint_positions_rad, dtype=float))
            except Exception:
                pass
        joint_targets = []
        for joint_name in self.joint_names:
            prim = stage.GetPrimAtPath(f"{self.joint_root_path}/{joint_name}")
            if not prim.IsValid():
                raise RuntimeError(f"Joint prim not found: {self.joint_root_path}/{joint_name}")
            state_attr = prim.GetAttribute("state:angular:physics:position")
            state_value = state_attr.Get() if state_attr.IsValid() else None
            if state_value is not None:
                joint_targets.append(float(state_value))
                continue
            attr = prim.GetAttribute("drive:angular:physics:targetPosition")
            value = attr.Get() if attr.IsValid() else None
            joint_targets.append(float(value) if value is not None else 0.0)
        return np.asarray(joint_targets, dtype=float)

    def read_stage_joint_positions_deg(self, stage) -> np.ndarray | None:
        articulation = self._get_articulation()
        joint_indices = self._get_articulation_joint_indices()
        if articulation is not None and joint_indices is not None:
            try:
                joint_positions_rad = articulation.get_joint_positions(joint_indices=joint_indices)
                if joint_positions_rad is not None:
                    return np.rad2deg(np.asarray(joint_positions_rad, dtype=float))
            except Exception:
                pass
        if stage is None:
            return None
        joint_positions = []
        for joint_name in self.joint_names:
            prim = stage.GetPrimAtPath(f"{self.joint_root_path}/{joint_name}")
            if not prim.IsValid():
                return None
            state_attr = prim.GetAttribute("state:angular:physics:position")
            state_value = state_attr.Get() if state_attr.IsValid() else None
            if state_value is None:
                return None
            joint_positions.append(float(state_value))
        return np.asarray(joint_positions, dtype=float)

    def write_joint_targets_deg(self, stage, joint_targets_deg: np.ndarray) -> None:
        articulation = self._get_articulation()
        joint_indices = self._get_articulation_joint_indices()
        if articulation is not None and joint_indices is not None:
            try:
                from isaacsim.core.utils.types import ArticulationAction
                articulation.apply_action(
                    ArticulationAction(
                        joint_positions=np.deg2rad(np.asarray(joint_targets_deg, dtype=float)),
                        joint_indices=joint_indices,
                    )
                )
                return
            except Exception:
                pass
        for joint_name, target_deg in zip(self.joint_names, joint_targets_deg, strict=True):
            prim = stage.GetPrimAtPath(f"{self.joint_root_path}/{joint_name}")
            if not prim.IsValid():
                raise RuntimeError(f"Joint prim not found: {self.joint_root_path}/{joint_name}")
            attr = prim.GetAttribute("drive:angular:physics:targetPosition")
            if not attr.IsValid():
                raise RuntimeError(f"Joint drive target attribute missing on {self.joint_root_path}/{joint_name}")
            attr.Set(float(target_deg))

    def write_joint_state_deg(self, stage, joint_positions_deg: np.ndarray) -> None:
        articulation = self._get_articulation()
        joint_indices = self._get_articulation_joint_indices()
        if articulation is not None and joint_indices is not None:
            try:
                articulation.set_joint_positions(np.deg2rad(np.asarray(joint_positions_deg, dtype=float)), joint_indices=joint_indices)
                return
            except Exception:
                pass
        for joint_name, position_deg in zip(self.joint_names, joint_positions_deg, strict=True):
            prim = stage.GetPrimAtPath(f"{self.joint_root_path}/{joint_name}")
            if not prim.IsValid():
                continue
            pos_attr = prim.GetAttribute("state:angular:physics:position")
            vel_attr = prim.GetAttribute("state:angular:physics:velocity")
            if pos_attr.IsValid():
                pos_attr.Set(float(position_deg))
            if vel_attr.IsValid():
                vel_attr.Set(0.0)
