#!/usr/bin/env python3

from __future__ import annotations

import numpy as np


class HopeJrStageIo:
    def __init__(
        self,
        *,
        articulation_root_path: str,
        joint_root_path: str,
        end_effector_path: str,
        joint_names: list[str] | tuple[str, ...],
    ):
        self.articulation_root_path = articulation_root_path
        self.joint_root_path = joint_root_path.rstrip("/")
        self.end_effector_path = end_effector_path
        self.joint_names = tuple(joint_names)
        self._articulation = None
        self._articulation_joint_indices = None
        self.last_stage_dls_debug = {}

    def get_stage(self):
        try:
            import omni.usd
        except ImportError:
            return None
        return omni.usd.get_context().get_stage()

    def _to_numpy(self, value):
        if value is None:
            return None
        if hasattr(value, "numpy"):
            value = value.numpy()
        elif hasattr(value, "cpu"):
            value = value.cpu().numpy()
        return np.asarray(value, dtype=float)

    def read_prim_pose(self, stage, prim_path: str) -> np.ndarray | None:
        if stage is None:
            return None
        try:
            from pxr import UsdGeom
        except ImportError:
            return None
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return None
        try:
            xform_cache = UsdGeom.XformCache()
            world_transform = xform_cache.GetLocalToWorldTransform(prim)
            return np.array(world_transform, dtype=float).T
        except Exception:
            return None

    def read_stage_end_effector_pose(self, stage) -> np.ndarray | None:
        return self.read_prim_pose(stage, self.end_effector_path)

    def read_stage_end_effector_position(self, stage) -> np.ndarray | None:
        pose = self.read_stage_end_effector_pose(stage)
        if pose is None:
            return None
        return pose[:3, 3].copy()

    def _set_stage_dls_debug(self, **kwargs) -> None:
        self.last_stage_dls_debug = dict(kwargs)

    def _get_articulation(self):
        try:
            from isaacsim.core.prims import Articulation
        except ImportError:
            self._set_stage_dls_debug(reason="articulation_import_failed")
            return None
        stage = self.get_stage()
        if stage is None:
            self._set_stage_dls_debug(reason="stage_unavailable")
            return None
        articulation_prim = stage.GetPrimAtPath(self.articulation_root_path)
        if not articulation_prim.IsValid():
            self._articulation = None
            self._articulation_joint_indices = None
            self._set_stage_dls_debug(reason="articulation_prim_invalid", articulation_root_path=self.articulation_root_path)
            return None
        if self._articulation is None:
            self._articulation = Articulation(self.articulation_root_path, reset_xform_properties=False)
            try:
                self._articulation.initialize()
            except Exception as exc:
                self._articulation = None
                self._set_stage_dls_debug(reason="articulation_initialize_failed", error=str(exc))
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
            except Exception as exc:
                self._set_stage_dls_debug(reason="joint_index_lookup_failed", error=str(exc), joint_names=list(self.joint_names))
                return None
        return self._articulation_joint_indices

    def stage_to_model_joint_positions_deg(self, joint_positions_deg: np.ndarray) -> np.ndarray:
        return np.asarray(joint_positions_deg, dtype=float)

    def model_to_stage_joint_positions_deg(self, joint_positions_deg: np.ndarray) -> np.ndarray:
        return np.asarray(joint_positions_deg, dtype=float)

    def read_current_joint_targets_deg(self, stage) -> np.ndarray:
        articulation = self._get_articulation()
        joint_indices = self._get_articulation_joint_indices()
        if articulation is not None and joint_indices is not None:
            try:
                joint_positions_rad = articulation.get_joint_positions(joint_indices=joint_indices)
                if joint_positions_rad is not None:
                    joint_positions_rad = np.asarray(joint_positions_rad, dtype=float)
                    if joint_positions_rad.ndim == 2:
                        joint_positions_rad = joint_positions_rad[0]
                    return np.rad2deg(joint_positions_rad)
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
                    joint_positions_rad = np.asarray(joint_positions_rad, dtype=float)
                    if joint_positions_rad.ndim == 2:
                        joint_positions_rad = joint_positions_rad[0]
                    return np.rad2deg(joint_positions_rad)
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

    def compute_end_effector_jacobian(self, stage, *, body_name: str = "PalmBody") -> np.ndarray | None:
        articulation = self._get_articulation()
        joint_indices = self._get_articulation_joint_indices()
        if articulation is None or joint_indices is None:
            if not self.last_stage_dls_debug:
                self._set_stage_dls_debug(reason="articulation_or_joint_indices_unavailable")
            return None
        try:
            jacobians = self._to_numpy(articulation.get_jacobians())
            if jacobians is None:
                self._set_stage_dls_debug(reason="jacobians_unavailable")
                return None
            if jacobians.ndim != 4 or jacobians.shape[0] == 0:
                self._set_stage_dls_debug(reason="jacobian_shape_invalid", jacobians_shape=list(jacobians.shape))
                return None
            jacobian = jacobians[0]
            body_index = articulation.get_body_index(body_name)
            body_names = list(articulation.body_names)
            jacobian_row = body_index
            if jacobian.shape[0] == len(body_names) - 1:
                if body_index == 0:
                    self._set_stage_dls_debug(reason="body_index_zero_for_fixed_base", body_name=body_name, body_index=int(body_index), body_names=body_names, jacobian_shape=list(jacobian.shape))
                    return None
                jacobian_row = body_index - 1
            if jacobian_row < 0 or jacobian_row >= jacobian.shape[0]:
                self._set_stage_dls_debug(reason="jacobian_row_out_of_range", body_name=body_name, body_index=int(body_index), jacobian_row=int(jacobian_row), body_names=body_names, jacobian_shape=list(jacobian.shape))
                return None
            body_jacobian = np.asarray(jacobian[jacobian_row][:, joint_indices], dtype=float)
        except Exception as exc:
            self._set_stage_dls_debug(reason="jacobian_fetch_failed", error=str(exc), body_name=body_name)
            return None

        body_pose = self.read_prim_pose(stage, f"{self.articulation_root_path}/{body_name}")
        end_effector_pose = self.read_stage_end_effector_pose(stage)
        debug_payload = {
            "reason": "ok",
            "body_name": body_name,
            "body_index": int(body_index),
            "jacobian_row": int(jacobian_row),
            "body_names": body_names,
            "jacobian_shape": list(jacobian.shape),
            "joint_indices": joint_indices.tolist(),
            "body_pose_available": body_pose is not None,
            "end_effector_pose_available": end_effector_pose is not None,
        }
        if body_pose is None or end_effector_pose is None:
            self._set_stage_dls_debug(**debug_payload)
            return body_jacobian

        offset_world = end_effector_pose[:3, 3] - body_pose[:3, 3]
        skew = np.array(
            [
                [0.0, -offset_world[2], offset_world[1]],
                [offset_world[2], 0.0, -offset_world[0]],
                [-offset_world[1], offset_world[0], 0.0],
            ],
            dtype=float,
        )
        linear = body_jacobian[:3, :] - skew @ body_jacobian[3:, :]
        angular = body_jacobian[3:, :]
        debug_payload["offset_world"] = offset_world.tolist()
        self._set_stage_dls_debug(**debug_payload)
        return np.vstack([linear, angular])

    def write_joint_targets_deg(self, stage, joint_targets_deg: np.ndarray) -> None:
        articulation = self._get_articulation()
        joint_indices = self._get_articulation_joint_indices()
        if articulation is None or joint_indices is None:
            raise RuntimeError("Articulation unavailable for write_joint_targets_deg")
        articulation.set_joint_position_targets(
            positions=np.deg2rad(np.asarray(joint_targets_deg, dtype=float))[None, :],
            joint_indices=joint_indices,
        )

    def write_joint_state_deg(self, stage, joint_positions_deg: np.ndarray) -> None:
        articulation = self._get_articulation()
        joint_indices = self._get_articulation_joint_indices()
        if articulation is None or joint_indices is None:
            raise RuntimeError("Articulation unavailable for write_joint_state_deg")
        articulation.set_joint_positions(np.deg2rad(np.asarray(joint_positions_deg, dtype=float))[None, :], joint_indices=joint_indices)
