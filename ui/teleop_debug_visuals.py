#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

try:
    from isaacsim.core.api.objects.ground_plane import GroundPlane
except ImportError:
    GroundPlane = None


class TeleopDebugVisuals:
    def __init__(self, *, teleop_debug_root: str, enabled: bool = True):
        self.teleop_debug_root = teleop_debug_root.rstrip("/")
        self.enabled = bool(enabled)

    def _set_display_color(self, prim, sdf, gf, color: tuple[float, float, float]) -> None:
        display_attr = prim.GetAttribute("primvars:displayColor")
        if not display_attr.IsValid():
            display_attr = prim.CreateAttribute("primvars:displayColor", sdf.ValueTypeNames.Color3fArray)
        display_attr.Set([gf.Vec3f(*color)])

    def _set_translate(self, prim, sdf, gf, position) -> None:
        translate_attr = prim.GetAttribute("xformOp:translate")
        if not translate_attr.IsValid():
            translate_attr = prim.CreateAttribute("xformOp:translate", sdf.ValueTypeNames.Double3)
        translate_attr.Set(gf.Vec3d(*[float(v) for v in position]))

    def _set_orient(self, prim, sdf, gf, quat_wxyz: tuple[float, float, float, float]) -> None:
        orient_attr = prim.GetAttribute("xformOp:orient")
        if not orient_attr.IsValid():
            orient_attr = prim.CreateAttribute("xformOp:orient", sdf.ValueTypeNames.Quatf)
        orient_attr.Set(gf.Quatf(*quat_wxyz))

    def _set_order(self, prim, order: list[str]) -> None:
        order_attr = prim.GetAttribute("xformOpOrder")
        order_attr.Set(order)

    def _define_marker_sphere(self, stage, usd_geom, sdf, gf, name: str, position, color, radius: float) -> None:
        sphere_path = f"{self.teleop_debug_root}/{name}"
        sphere = usd_geom.Sphere.Define(stage, sphere_path)
        prim = sphere.GetPrim()
        self._set_display_color(prim, sdf, gf, color)
        sphere.GetRadiusAttr().Set(radius)
        self._set_translate(prim, sdf, gf, position)
        self._set_order(prim, ["xformOp:translate"])

    def _define_target_cross(self, stage, usd_geom, sdf, gf, sim_target_position, waiting_for_anchor: bool) -> None:
        root_path = f"{self.teleop_debug_root}/SimTargetCross"
        root = stage.DefinePrim(root_path, "Xform")
        self._set_translate(root, sdf, gf, sim_target_position)
        self._set_order(root, ["xformOp:translate"])

        live_color = (0.9, 0.1, 0.1)
        wait_color = (0.0, 1.0, 0.0)
        bar_color = wait_color if waiting_for_anchor else live_color
        bar_specs = [
            ("XBar", bar_color, Rotation.from_euler("y", 90.0, degrees=True)),
            ("YBar", bar_color, Rotation.from_euler("x", 90.0, degrees=True)),
            ("ZBar", bar_color, Rotation.identity()),
        ]

        for name, color, rotation in bar_specs:
            cylinder = usd_geom.Cylinder.Define(stage, f"{root_path}/{name}")
            prim = cylinder.GetPrim()
            cylinder.GetRadiusAttr().Set(0.0012)
            cylinder.GetHeightAttr().Set(0.05)
            self._set_display_color(prim, sdf, gf, color)
            quat_xyzw = rotation.as_quat()
            quat_wxyz = (float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2]))
            self._set_orient(prim, sdf, gf, quat_wxyz)
            self._set_order(prim, ["xformOp:orient"])

    def _ensure_ground_plane(self) -> None:
        if GroundPlane is None:
            return
        try:
            GroundPlane(prim_path="/World/GroundPlane", z_position=-0.65)
        except Exception:
            pass

    def _define_scene_backdrop(self, stage, usd_geom, sdf, gf, reference_position) -> None:
        root_path = "/World/TeleopBackdrop"
        prim = stage.GetPrimAtPath(root_path)
        if prim.IsValid():
            stage.RemovePrim(root_path)

    def update(
        self,
        stage,
        *,
        quest_anchor_position: np.ndarray,
        quest_current_position: np.ndarray,
        quest_mapped_position: np.ndarray,
        sim_target_position: np.ndarray,
        reference_position: np.ndarray | None = None,
        actual_end_effector_position: np.ndarray | None = None,
        actual_end_effector_pose: np.ndarray | None = None,
        waiting_for_anchor: bool = False,
    ) -> None:
        if not self.enabled or stage is None:
            return
        try:
            from pxr import Gf, Sdf, UsdGeom
        except ImportError:
            return

        self._ensure_ground_plane()
        stage.DefinePrim(self.teleop_debug_root, "Xform")
        sim_target_color = (0.0, 1.0, 0.0) if waiting_for_anchor else (1.0, 0.0, 0.0)
        if reference_position is not None:
            self._define_scene_backdrop(stage, UsdGeom, Sdf, Gf, reference_position)
        self._define_marker_sphere(stage, UsdGeom, Sdf, Gf, "QuestMapped", quest_mapped_position, (1.0, 0.5, 0.0), 0.0045)
        self._define_marker_sphere(stage, UsdGeom, Sdf, Gf, "SimTarget", sim_target_position, sim_target_color, 0.0065)
        self._define_target_cross(stage, UsdGeom, Sdf, Gf, sim_target_position, waiting_for_anchor=waiting_for_anchor)

        if actual_end_effector_pose is None:
            return

        arrow_root_path = f"{self.teleop_debug_root}/ActualEndEffectorArrow"
        arrow_root = stage.DefinePrim(arrow_root_path, "Xform")
        arrow_rotation = actual_end_effector_pose[:3, :3]
        quat_xyzw = Rotation.from_matrix(arrow_rotation).as_quat()
        quat_wxyz = [float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])]
        self._set_translate(arrow_root, Sdf, Gf, actual_end_effector_pose[:3, 3])
        self._set_orient(arrow_root, Sdf, Gf, tuple(quat_wxyz))
        self._set_order(arrow_root, ["xformOp:translate", "xformOp:orient"])

        shaft = UsdGeom.Cylinder.Define(stage, f"{arrow_root_path}/Shaft")
        shaft_prim = shaft.GetPrim()
        shaft.GetRadiusAttr().Set(0.0028)
        shaft.GetHeightAttr().Set(0.03)
        self._set_display_color(shaft_prim, Sdf, Gf, (0.1, 0.5, 1.0))
        self._set_translate(shaft_prim, Sdf, Gf, (0.0, 0.0, 0.015))
        self._set_order(shaft_prim, ["xformOp:translate"])

        tip = UsdGeom.Cone.Define(stage, f"{arrow_root_path}/Tip")
        tip_prim = tip.GetPrim()
        tip.GetRadiusAttr().Set(0.005)
        tip.GetHeightAttr().Set(0.014)
        self._set_display_color(tip_prim, Sdf, Gf, (0.1, 0.5, 1.0))
        self._set_translate(tip_prim, Sdf, Gf, (0.0, 0.0, 0.032))
        self._set_order(tip_prim, ["xformOp:translate"])
