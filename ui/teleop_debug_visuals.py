#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation
import carb

try:
    from isaacsim.core.api.objects.ground_plane import GroundPlane
except ImportError:
    GroundPlane = None
try:
    from isaacsim.core.api.objects import DynamicCuboid
except ImportError:
    DynamicCuboid = None
try:
    from omni.isaac.core.utils.nucleus import get_assets_root_path
except ImportError:
    try:
        from omni.isaac.nucleus import get_assets_root_path
    except ImportError:
        get_assets_root_path = None


class TeleopDebugVisuals:
    def __init__(self, *, teleop_debug_root: str, enabled: bool = True):
        self.teleop_debug_root = teleop_debug_root.rstrip("/")
        self.enabled = bool(enabled)
        self._table_asset_path: str | None = None
        self._table_asset_resolved = False
        self._table_spawned = False
        self._cube_spawned = False

    def _set_visibility(self, prim, usd_geom, visible: bool) -> None:
        imageable = usd_geom.Imageable(prim)
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()

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

    def _clear_prim(self, stage, path: str) -> None:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            stage.RemovePrim(path)

    def _ensure_table_asset(self, stage, sdf, gf) -> None:
        if not self._table_asset_resolved:
            asset_root = None
            if get_assets_root_path is not None:
                try:
                    asset_root = get_assets_root_path()
                except Exception:
                    asset_root = None
            if not asset_root:
                try:
                    asset_root = carb.settings.get_settings().get("/persistent/isaac/asset_root/cloud")
                except Exception:
                    asset_root = None
            if asset_root:
                self._table_asset_path = f"{asset_root}/Isaac/Props/Mounts/SeattleLabTable/table_instanceable.usd"
            self._table_asset_resolved = True
        if not self._table_asset_path:
            return
        prim_path = "/World/Table"
        table = stage.GetPrimAtPath(prim_path)
        if not table.IsValid():
            table = stage.DefinePrim(prim_path, "Xform")
        if not self._table_spawned:
            refs = table.GetReferences()
            refs.ClearReferences()
            refs.AddReference(self._table_asset_path)
            self._table_spawned = True

        self._set_translate(table, sdf, gf, (0.05584, 0.09835, -0.33866))
        self._set_orient(table, sdf, gf, (1.0, 0.0, 0.0, 0.0))
        scale_attr = table.GetAttribute("xformOp:scale")
        if not scale_attr.IsValid():
            scale_attr = table.CreateAttribute("xformOp:scale", sdf.ValueTypeNames.Double3)
        scale_attr.Set(gf.Vec3d(1.0, 1.0, 1.0))
        self._set_order(table, ["xformOp:translate", "xformOp:orient", "xformOp:scale"])

    def _ensure_table_cube(self, stage) -> None:
        if DynamicCuboid is None:
            return
        prim_path = "/World/TableCube"
        if stage.GetPrimAtPath(prim_path).IsValid():
            self._cube_spawned = True
            return
        if self._cube_spawned:
            return
        DynamicCuboid(
            prim_path=prim_path,
            name="table_cube",
            position=np.array([0.20529, 0.14001, -0.32166]),
            size=1.0,
            scale=np.array([0.04, 0.04, 0.04]),
            mass=0.05,
            density=400.0,
            color=np.array([0.82, 0.18, 0.18]),
        )
        self._cube_spawned = True

    def _define_segment(self, stage, usd_geom, sdf, gf, path: str, start, end, color, radius: float) -> None:
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length <= 1e-9:
            return
        midpoint = 0.5 * (start + end)
        rotation, _ = Rotation.align_vectors([direction / length], [np.array([0.0, 0.0, 1.0])])
        quat_xyzw = rotation.as_quat()
        quat_wxyz = (float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2]))
        cylinder = usd_geom.Cylinder.Define(stage, path)
        prim = cylinder.GetPrim()
        cylinder.GetRadiusAttr().Set(radius)
        cylinder.GetHeightAttr().Set(length)
        self._set_display_color(prim, sdf, gf, color)
        self._set_translate(prim, sdf, gf, midpoint)
        self._set_orient(prim, sdf, gf, quat_wxyz)
        self._set_order(prim, ["xformOp:translate", "xformOp:orient"])

    def _define_polyline(self, stage, usd_geom, sdf, gf, root_path: str, points: list[np.ndarray], color, radius: float) -> None:
        for idx in range(len(points) - 1):
            self._define_segment(
                stage,
                usd_geom,
                sdf,
                gf,
                f"{root_path}/Seg{idx}",
                points[idx],
                points[idx + 1],
                color,
                radius,
            )

    def _define_frame_axes(self, stage, usd_geom, sdf, gf, root_path: str, transform: np.ndarray, *, axis_length: float, radius: float, colors) -> None:
        origin = np.asarray(transform[:3, 3], dtype=float)
        basis = np.asarray(transform[:3, :3], dtype=float)
        sphere = usd_geom.Sphere.Define(stage, f"{root_path}/Origin")
        prim = sphere.GetPrim()
        self._set_display_color(prim, sdf, gf, colors[0])
        sphere.GetRadiusAttr().Set(radius * 1.8)
        self._set_translate(prim, sdf, gf, origin)
        self._set_order(prim, ["xformOp:translate"])
        labels = ['X', 'Y', 'Z']
        for idx, label in enumerate(labels):
            end = origin + basis[:, idx] * axis_length
            self._define_segment(stage, usd_geom, sdf, gf, f"{root_path}/{label}", origin, end, colors[idx], radius)

    def _define_pitch_frames(self, stage, usd_geom, sdf, gf, pitch_visual: dict[str, np.ndarray] | None) -> None:
        root_path = f"{self.teleop_debug_root}/PitchFrames"
        if not isinstance(pitch_visual, dict):
            root_prim = stage.GetPrimAtPath(root_path)
            if root_prim.IsValid():
                self._set_visibility(root_prim, usd_geom, False)
            return
        parent_frame = np.asarray(pitch_visual.get('parent_frame'), dtype=float)
        child_frame = np.asarray(pitch_visual.get('child_frame'), dtype=float)
        child_frame_raw = np.asarray(pitch_visual.get('child_frame_raw'), dtype=float)
        axis_world = np.asarray(pitch_visual.get('axis_world'), dtype=float)
        root_prim = stage.DefinePrim(root_path, 'Xform')
        self._set_visibility(root_prim, usd_geom, True)
        self._define_frame_axes(stage, usd_geom, sdf, gf, f"{root_path}/Parent", parent_frame, axis_length=0.035, radius=0.0012, colors=((1.0, 0.72, 0.05), (1.0, 0.16, 0.06), (1.0, 0.92, 0.08)))
        self._define_frame_axes(stage, usd_geom, sdf, gf, f"{root_path}/Child", child_frame, axis_length=0.028, radius=0.0009, colors=((0.3, 0.8, 1.0), (0.8, 0.3, 1.0), (0.6, 0.9, 1.0)))
        parent_origin = np.asarray(parent_frame[:3, 3], dtype=float)
        child_origin_raw = np.asarray(child_frame_raw[:3, 3], dtype=float)
        child_origin_preview = np.asarray(child_frame[:3, 3], dtype=float)
        self._define_marker_sphere(stage, usd_geom, sdf, gf, f"PitchFrames/ParentOrigin", parent_origin, (1.0, 0.8, 0.2), 0.0032)
        self._define_marker_sphere(stage, usd_geom, sdf, gf, f"PitchFrames/ChildOriginRaw", child_origin_raw, (0.1, 0.9, 1.0), 0.0024)
        self._define_segment(stage, usd_geom, sdf, gf, f"{root_path}/ChildPreviewLink", child_origin_raw, child_origin_preview, (0.8, 0.95, 1.0), 0.0007)
        self._define_segment(stage, usd_geom, sdf, gf, f"{root_path}/ParentChildRawLink", parent_origin, child_origin_raw, (1.0, 1.0, 1.0), 0.0005)
        axis_norm = float(np.linalg.norm(axis_world))
        if axis_norm > 1e-9:
            axis_dir = axis_world / axis_norm
            self._define_segment(stage, usd_geom, sdf, gf, f"{root_path}/PitchAxis", parent_origin - axis_dir * 0.03, parent_origin + axis_dir * 0.03, (1.0, 0.0, 1.0), 0.0018)
            ref = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(float(np.dot(ref, axis_dir))) > 0.9:
                ref = np.array([1.0, 0.0, 0.0], dtype=float)
            tangent_a = np.cross(axis_dir, ref)
            tangent_a = tangent_a / float(np.linalg.norm(tangent_a))
            tangent_b = np.cross(axis_dir, tangent_a)
            arc_radius = 0.054
            arc_center = parent_origin + tangent_b * 0.036
            angles = np.linspace(-0.25 * np.pi, 0.65 * np.pi, 10)
            arc_points = [
                arc_center + tangent_a * (np.cos(theta) * arc_radius) + tangent_b * (np.sin(theta) * arc_radius)
                for theta in angles
            ]
            self._define_polyline(stage, usd_geom, sdf, gf, f"{root_path}/PitchArc", arc_points, (1.0, 0.0, 1.0), 0.001)
            arrow_tip = arc_points[-1]
            arrow_back = arc_points[-2]
            arrow_dir = arrow_tip - arrow_back
            arrow_dir = arrow_dir / max(float(np.linalg.norm(arrow_dir)), 1e-9)
            arrow_size = 0.024
            head_left = arrow_tip - arrow_dir * arrow_size + tangent_b * (arrow_size * 0.45)
            head_right = arrow_tip - arrow_dir * arrow_size - tangent_b * (arrow_size * 0.45)
            self._define_segment(stage, usd_geom, sdf, gf, f"{root_path}/PitchArcHeadL", arrow_tip, head_left, (1.0, 0.0, 1.0), 0.0011)
            self._define_segment(stage, usd_geom, sdf, gf, f"{root_path}/PitchArcHeadR", arrow_tip, head_right, (1.0, 0.0, 1.0), 0.0011)

    def _ensure_ground_plane(self, stage=None, usd_geom=None, sdf=None, gf=None) -> None:
        if GroundPlane is not None:
            try:
                GroundPlane(prim_path="/World/GroundPlane", z_position=-1.05)
            except Exception:
                pass
        if stage is None or usd_geom is None or sdf is None or gf is None:
            return
        root_path = "/World/TeleopGroundVisual"
        stage.DefinePrim(root_path, "Xform")
        dark = gf.Vec3f(0.16, 0.18, 0.22)
        light = gf.Vec3f(0.28, 0.31, 0.36)
        tile_half = 0.12
        z = -1.049
        self._create_checker_mesh(stage, usd_geom, sdf, f"{root_path}/Checker0", z=z, tile_half=tile_half, xdim=40, ydim=40, checker_mod=0, color=dark)
        self._create_checker_mesh(stage, usd_geom, sdf, f"{root_path}/Checker1", z=z, tile_half=tile_half, xdim=40, ydim=40, checker_mod=1, color=light)
        wall_color = gf.Vec3f(0.10, 0.12, 0.16)
        side_color = gf.Vec3f(0.12, 0.14, 0.18)
        stripe_color = gf.Vec3f(0.22, 0.24, 0.30)
        self._define_visual_wall(stage, usd_geom, sdf, gf, f"{root_path}/BackWall", position=(0.0, 1.10, 0.1), scale=(4.25, 0.04, 1.4), color=wall_color)
        self._define_visual_wall(stage, usd_geom, sdf, gf, f"{root_path}/BackWallStripeV0", position=(-1.2, 1.075, 0.1), scale=(0.18, 0.01, 1.4), color=stripe_color)
        self._define_visual_wall(stage, usd_geom, sdf, gf, f"{root_path}/BackWallStripeV1", position=(0.0, 1.075, 0.1), scale=(0.18, 0.01, 1.4), color=stripe_color)
        self._define_visual_wall(stage, usd_geom, sdf, gf, f"{root_path}/BackWallStripeV2", position=(1.2, 1.075, 0.1), scale=(0.18, 0.01, 1.4), color=stripe_color)
        self._define_visual_wall(stage, usd_geom, sdf, gf, f"{root_path}/BackWallStripeH0", position=(0.0, 1.075, -0.25), scale=(4.25, 0.01, 0.04), color=stripe_color)
        self._define_visual_wall(stage, usd_geom, sdf, gf, f"{root_path}/BackWallStripeH1", position=(0.0, 1.075, 0.10), scale=(4.25, 0.01, 0.04), color=stripe_color)
        self._define_visual_wall(stage, usd_geom, sdf, gf, f"{root_path}/BackWallStripeH2", position=(0.0, 1.075, 0.45), scale=(4.25, 0.01, 0.04), color=stripe_color)
        self._define_visual_wall(stage, usd_geom, sdf, gf, f"{root_path}/SideWall", position=(1.52, 0.0, 0.1), scale=(0.04, 4.25, 1.4), color=side_color)
        self._define_visual_wall(stage, usd_geom, sdf, gf, f"{root_path}/SideWallStripeV0", position=(1.495, -1.2, 0.1), scale=(0.01, 0.18, 1.4), color=stripe_color)
        self._define_visual_wall(stage, usd_geom, sdf, gf, f"{root_path}/SideWallStripeV1", position=(1.495, 0.0, 0.1), scale=(0.01, 0.18, 1.4), color=stripe_color)
        self._define_visual_wall(stage, usd_geom, sdf, gf, f"{root_path}/SideWallStripeV2", position=(1.495, 1.2, 0.1), scale=(0.01, 0.18, 1.4), color=stripe_color)
        self._define_visual_wall(stage, usd_geom, sdf, gf, f"{root_path}/SideWallStripeH0", position=(1.495, 0.0, -0.25), scale=(0.01, 4.25, 0.04), color=stripe_color)
        self._define_visual_wall(stage, usd_geom, sdf, gf, f"{root_path}/SideWallStripeH1", position=(1.495, 0.0, 0.10), scale=(0.01, 4.25, 0.04), color=stripe_color)
        self._define_visual_wall(stage, usd_geom, sdf, gf, f"{root_path}/SideWallStripeH2", position=(1.495, 0.0, 0.45), scale=(0.01, 4.25, 0.04), color=stripe_color)

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
        show_pitch_frames: bool = False,
        pitch_visual: dict[str, np.ndarray] | None = None,
    ) -> None:
        if not self.enabled or stage is None:
            return
        try:
            from pxr import Gf, Sdf, UsdGeom
        except ImportError:
            return

        self._ensure_ground_plane(stage, UsdGeom, Sdf, Gf)
        self._ensure_table_asset(stage, Sdf, Gf)
        self._ensure_table_cube(stage)
        stage.DefinePrim(self.teleop_debug_root, "Xform")
        sim_target_color = (0.0, 1.0, 0.0) if waiting_for_anchor else (1.0, 0.0, 0.0)
        if reference_position is not None:
            self._define_scene_backdrop(stage, UsdGeom, Sdf, Gf, reference_position)
        self._define_marker_sphere(stage, UsdGeom, Sdf, Gf, "QuestMapped", quest_mapped_position, (1.0, 0.5, 0.0), 0.0045)
        self._define_marker_sphere(stage, UsdGeom, Sdf, Gf, "SimTarget", sim_target_position, sim_target_color, 0.0065)
        self._define_target_cross(stage, UsdGeom, Sdf, Gf, sim_target_position, waiting_for_anchor=waiting_for_anchor)
        if show_pitch_frames:
            self._define_pitch_frames(stage, UsdGeom, Sdf, Gf, pitch_visual)
        else:
            pitch_frames_prim = stage.GetPrimAtPath(f"{self.teleop_debug_root}/PitchFrames")
            if pitch_frames_prim.IsValid():
                self._set_visibility(pitch_frames_prim, UsdGeom, False)

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
