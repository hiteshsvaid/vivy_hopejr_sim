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

    def _set_shadow_flag(self, prim, sdf) -> None:
        attr = prim.GetAttribute("primvars:doNotCastShadows")
        if not attr.IsValid():
            attr = prim.CreateAttribute("primvars:doNotCastShadows", sdf.ValueTypeNames.Bool)
        attr.Set(True)

    def _create_checker_mesh(self, stage, usd_geom, sdf, path: str, *, z: float, tile_half: float, xdim: int, ydim: int, checker_mod: int, color) -> None:
        xdim_half = xdim / 2
        ydim_half = ydim / 2
        points = []
        counts = []
        indices = []
        for y in range(ydim):
            y_offset = (y - ydim_half) * tile_half * 2.0 + tile_half
            for x in range(xdim):
                x_offset = (x - xdim_half) * tile_half * 2.0 + tile_half
                if (x + y) % 2 != checker_mod:
                    continue
                base = len(points)
                points.extend([
                    (-tile_half + x_offset, -tile_half + y_offset, z),
                    ( tile_half + x_offset, -tile_half + y_offset, z),
                    ( tile_half + x_offset,  tile_half + y_offset, z),
                    (-tile_half + x_offset,  tile_half + y_offset, z),
                ])
                counts.append(4)
                indices.extend([base + 0, base + 1, base + 2, base + 3])

        mesh = usd_geom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr(counts)
        mesh.CreateFaceVertexIndicesAttr(indices)
        mesh.CreateDisplayColorAttr().Set([color])
        prim = mesh.GetPrim()
        prim.CreateAttribute("purpose", sdf.ValueTypeNames.Token).Set("render")
        self._set_shadow_flag(prim, sdf)

    def _define_visual_wall(self, stage, usd_geom, sdf, gf, path: str, *, position, scale, color) -> None:
        wall = usd_geom.Cube.Define(stage, path)
        wall.CreateSizeAttr(1.0)
        prim = wall.GetPrim()
        self._set_display_color(prim, sdf, gf, color)
        self._set_translate(prim, sdf, gf, position)
        scale_attr = prim.GetAttribute("xformOp:scale")
        if not scale_attr.IsValid():
            scale_attr = prim.CreateAttribute("xformOp:scale", sdf.ValueTypeNames.Double3)
        scale_attr.Set(gf.Vec3d(*[float(v) for v in scale]))
        self._set_order(prim, ["xformOp:translate", "xformOp:scale"])
        prim.CreateAttribute("purpose", sdf.ValueTypeNames.Token).Set("render")
        self._set_shadow_flag(prim, sdf)

    def _ensure_ground_plane(self, stage=None, usd_geom=None, sdf=None, gf=None) -> None:
        if GroundPlane is not None:
            try:
                GroundPlane(prim_path="/World/GroundPlane", z_position=-0.60)
            except Exception:
                pass
        if stage is None or usd_geom is None or sdf is None or gf is None:
            return
        root_path = "/World/TeleopGroundVisual"
        stage.DefinePrim(root_path, "Xform")
        dark = gf.Vec3f(0.16, 0.18, 0.22)
        light = gf.Vec3f(0.28, 0.31, 0.36)
        tile_half = 0.12
        z = -0.599
        self._create_checker_mesh(stage, usd_geom, sdf, f"{root_path}/Checker0", z=z, tile_half=tile_half, xdim=40, ydim=40, checker_mod=0, color=dark)
        self._create_checker_mesh(stage, usd_geom, sdf, f"{root_path}/Checker1", z=z, tile_half=tile_half, xdim=40, ydim=40, checker_mod=1, color=light)
        wall_color = gf.Vec3f(0.10, 0.12, 0.16)
        side_color = gf.Vec3f(0.12, 0.14, 0.18)
        stripe_color = gf.Vec3f(0.22, 0.24, 0.30)
        self._define_visual_wall(stage, usd_geom, sdf, gf, f"{root_path}/BackWall", position=(0.0, 1.47, 0.1), scale=(4.25, 0.04, 1.4), color=wall_color)
        self._define_visual_wall(stage, usd_geom, sdf, gf, f"{root_path}/BackWallStripeV0", position=(-1.2, 1.445, 0.1), scale=(0.18, 0.01, 1.4), color=stripe_color)
        self._define_visual_wall(stage, usd_geom, sdf, gf, f"{root_path}/BackWallStripeV1", position=(0.0, 1.445, 0.1), scale=(0.18, 0.01, 1.4), color=stripe_color)
        self._define_visual_wall(stage, usd_geom, sdf, gf, f"{root_path}/BackWallStripeV2", position=(1.2, 1.445, 0.1), scale=(0.18, 0.01, 1.4), color=stripe_color)
        self._define_visual_wall(stage, usd_geom, sdf, gf, f"{root_path}/BackWallStripeH0", position=(0.0, 1.445, -0.25), scale=(4.25, 0.01, 0.04), color=stripe_color)
        self._define_visual_wall(stage, usd_geom, sdf, gf, f"{root_path}/BackWallStripeH1", position=(0.0, 1.445, 0.10), scale=(4.25, 0.01, 0.04), color=stripe_color)
        self._define_visual_wall(stage, usd_geom, sdf, gf, f"{root_path}/BackWallStripeH2", position=(0.0, 1.445, 0.45), scale=(4.25, 0.01, 0.04), color=stripe_color)
        self._define_visual_wall(stage, usd_geom, sdf, gf, f"{root_path}/SideWall", position=(-1.344, 0.0, 0.1), scale=(0.04, 4.25, 1.4), color=side_color)
        self._define_visual_wall(stage, usd_geom, sdf, gf, f"{root_path}/SideWallStripeV0", position=(-1.319, -1.2, 0.1), scale=(0.01, 0.18, 1.4), color=stripe_color)
        self._define_visual_wall(stage, usd_geom, sdf, gf, f"{root_path}/SideWallStripeV1", position=(-1.319, 0.0, 0.1), scale=(0.01, 0.18, 1.4), color=stripe_color)
        self._define_visual_wall(stage, usd_geom, sdf, gf, f"{root_path}/SideWallStripeV2", position=(-1.319, 1.2, 0.1), scale=(0.01, 0.18, 1.4), color=stripe_color)
        self._define_visual_wall(stage, usd_geom, sdf, gf, f"{root_path}/SideWallStripeH0", position=(-1.319, 0.0, -0.25), scale=(0.01, 4.25, 0.04), color=stripe_color)
        self._define_visual_wall(stage, usd_geom, sdf, gf, f"{root_path}/SideWallStripeH1", position=(-1.319, 0.0, 0.10), scale=(0.01, 4.25, 0.04), color=stripe_color)
        self._define_visual_wall(stage, usd_geom, sdf, gf, f"{root_path}/SideWallStripeH2", position=(-1.319, 0.0, 0.45), scale=(0.01, 4.25, 0.04), color=stripe_color)

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

        self._ensure_ground_plane(stage, UsdGeom, Sdf, Gf)
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
