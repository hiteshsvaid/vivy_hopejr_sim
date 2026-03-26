#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


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

    def _define_scene_backdrop(self, stage, usd_geom, sdf, gf, reference_position) -> None:
        root_path = "/World/TeleopBackdrop"
        root = stage.DefinePrim(root_path, "Xform")
        anchor = np.asarray(reference_position, dtype=float)
        backdrop_origin = anchor + np.array([0.16, 0.0, -0.09], dtype=float)
        self._set_translate(root, sdf, gf, backdrop_origin)
        self._set_order(root, ["xformOp:translate"])

        wall = usd_geom.Cube.Define(stage, f"{root_path}/Wall")
        wall_prim = wall.GetPrim()
        wall.GetSizeAttr().Set(1.0)
        self._set_display_color(wall_prim, sdf, gf, (0.72, 0.72, 0.72))
        wall_translate = wall_prim.GetAttribute("xformOp:translate")
        if not wall_translate.IsValid():
            wall_translate = wall_prim.CreateAttribute("xformOp:translate", sdf.ValueTypeNames.Double3)
        wall_translate.Set(gf.Vec3d(0.06, 0.0, 0.12))
        wall_scale = wall_prim.GetAttribute("xformOp:scale")
        if not wall_scale.IsValid():
            wall_scale = wall_prim.CreateAttribute("xformOp:scale", sdf.ValueTypeNames.Double3)
        wall_scale.Set(gf.Vec3d(0.004, 0.48, 0.32))
        self._set_order(wall_prim, ["xformOp:translate", "xformOp:scale"])

        wall_grid_root_path = f"{root_path}/WallGrid"
        wall_grid_root = stage.DefinePrim(wall_grid_root_path, "Xform")
        self._set_order(wall_grid_root, [])

        wall_y_positions = (-0.20, -0.10, 0.0, 0.10, 0.20)
        wall_z_positions = (-0.12, -0.04, 0.04, 0.12)

        for idx, y in enumerate(wall_y_positions):
            line = usd_geom.Cube.Define(stage, f"{wall_grid_root_path}/Vertical{idx}")
            prim = line.GetPrim()
            line.GetSizeAttr().Set(1.0)
            self._set_display_color(prim, sdf, gf, (0.92, 0.92, 0.92) if y == 0.0 else (0.80, 0.80, 0.80))
            translate = prim.GetAttribute("xformOp:translate")
            if not translate.IsValid():
                translate = prim.CreateAttribute("xformOp:translate", sdf.ValueTypeNames.Double3)
            translate.Set(gf.Vec3d(0.058, float(y), 0.12))
            scale = prim.GetAttribute("xformOp:scale")
            if not scale.IsValid():
                scale = prim.CreateAttribute("xformOp:scale", sdf.ValueTypeNames.Double3)
            scale.Set(gf.Vec3d(0.001, 0.0012, 0.16))
            self._set_order(prim, ["xformOp:translate", "xformOp:scale"])

        for idx, z in enumerate(wall_z_positions):
            line = usd_geom.Cube.Define(stage, f"{wall_grid_root_path}/Horizontal{idx}")
            prim = line.GetPrim()
            line.GetSizeAttr().Set(1.0)
            self._set_display_color(prim, sdf, gf, (0.92, 0.92, 0.92) if abs(z - 0.04) < 1e-9 else (0.80, 0.80, 0.80))
            translate = prim.GetAttribute("xformOp:translate")
            if not translate.IsValid():
                translate = prim.CreateAttribute("xformOp:translate", sdf.ValueTypeNames.Double3)
            translate.Set(gf.Vec3d(0.058, 0.0, float(z) + 0.12))
            scale = prim.GetAttribute("xformOp:scale")
            if not scale.IsValid():
                scale = prim.CreateAttribute("xformOp:scale", sdf.ValueTypeNames.Double3)
            scale.Set(gf.Vec3d(0.001, 0.24, 0.0012))
            self._set_order(prim, ["xformOp:translate", "xformOp:scale"])

        floor = usd_geom.Cube.Define(stage, f"{root_path}/Floor")
        floor_prim = floor.GetPrim()
        floor.GetSizeAttr().Set(1.0)
        self._set_display_color(floor_prim, sdf, gf, (0.58, 0.58, 0.58))
        floor_translate = floor_prim.GetAttribute("xformOp:translate")
        if not floor_translate.IsValid():
            floor_translate = floor_prim.CreateAttribute("xformOp:translate", sdf.ValueTypeNames.Double3)
        floor_translate.Set(gf.Vec3d(-0.10, 0.0, -0.04))
        floor_scale = floor_prim.GetAttribute("xformOp:scale")
        if not floor_scale.IsValid():
            floor_scale = floor_prim.CreateAttribute("xformOp:scale", sdf.ValueTypeNames.Double3)
        floor_scale.Set(gf.Vec3d(0.56, 0.44, 0.004))
        self._set_order(floor_prim, ["xformOp:translate", "xformOp:scale"])

        grid_root_path = f"{root_path}/FloorGrid"
        grid_root = stage.DefinePrim(grid_root_path, "Xform")
        self._set_order(grid_root, [])

        x_positions = (-0.38, -0.24, -0.10, 0.04, 0.18)
        y_positions = (-0.20, -0.10, 0.0, 0.10, 0.20)

        for idx, x in enumerate(x_positions):
            line = usd_geom.Cube.Define(stage, f"{grid_root_path}/XLine{idx}")
            prim = line.GetPrim()
            line.GetSizeAttr().Set(1.0)
            self._set_display_color(prim, sdf, gf, (0.90, 0.90, 0.90) if x == -0.10 else (0.78, 0.78, 0.78))
            translate = prim.GetAttribute("xformOp:translate")
            if not translate.IsValid():
                translate = prim.CreateAttribute("xformOp:translate", sdf.ValueTypeNames.Double3)
            translate.Set(gf.Vec3d(float(x), 0.0, -0.036))
            scale = prim.GetAttribute("xformOp:scale")
            if not scale.IsValid():
                scale = prim.CreateAttribute("xformOp:scale", sdf.ValueTypeNames.Double3)
            scale.Set(gf.Vec3d(0.0015, 0.44, 0.0015))
            self._set_order(prim, ["xformOp:translate", "xformOp:scale"])

        for idx, y in enumerate(y_positions):
            line = usd_geom.Cube.Define(stage, f"{grid_root_path}/YLine{idx}")
            prim = line.GetPrim()
            line.GetSizeAttr().Set(1.0)
            self._set_display_color(prim, sdf, gf, (0.90, 0.90, 0.90) if y == 0.0 else (0.78, 0.78, 0.78))
            translate = prim.GetAttribute("xformOp:translate")
            if not translate.IsValid():
                translate = prim.CreateAttribute("xformOp:translate", sdf.ValueTypeNames.Double3)
            translate.Set(gf.Vec3d(-0.10, float(y), -0.0355))
            scale = prim.GetAttribute("xformOp:scale")
            if not scale.IsValid():
                scale = prim.CreateAttribute("xformOp:scale", sdf.ValueTypeNames.Double3)
            scale.Set(gf.Vec3d(0.28, 0.0015, 0.0015))
            self._set_order(prim, ["xformOp:translate", "xformOp:scale"])

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
