#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


class TeleopDebugVisuals:
    def __init__(self, *, teleop_debug_root: str, enabled: bool = True):
        self.teleop_debug_root = teleop_debug_root.rstrip("/")
        self.enabled = bool(enabled)

    def update(
        self,
        stage,
        *,
        quest_anchor_position: np.ndarray,
        quest_current_position: np.ndarray,
        quest_mapped_position: np.ndarray,
        sim_target_position: np.ndarray,
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
        visuals = [
            ("QuestMapped", quest_mapped_position, (1.0, 0.5, 0.0), 0.004),
            ("SimTarget", sim_target_position, sim_target_color, 0.005),
        ]
        for name, position, color, radius in visuals:
            sphere_path = f"{self.teleop_debug_root}/{name}"
            sphere = UsdGeom.Sphere.Define(stage, sphere_path)
            prim = sphere.GetPrim()
            display_attr = prim.GetAttribute("primvars:displayColor")
            if not display_attr.IsValid():
                display_attr = prim.CreateAttribute("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray)
            display_attr.Set([Gf.Vec3f(*color)])
            sphere.GetRadiusAttr().Set(radius)
            translate_attr = prim.GetAttribute("xformOp:translate")
            if not translate_attr.IsValid():
                translate_attr = prim.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3)
            translate_attr.Set(Gf.Vec3d(*[float(v) for v in position]))
            order_attr = prim.GetAttribute("xformOpOrder")
            if not order_attr.IsValid() or not order_attr.Get():
                order_attr.Set(["xformOp:translate"])

        if actual_end_effector_pose is None:
            return

        arrow_root_path = f"{self.teleop_debug_root}/ActualEndEffectorArrow"
        arrow_root = stage.DefinePrim(arrow_root_path, "Xform")
        arrow_rotation = actual_end_effector_pose[:3, :3]
        quat_xyzw = Rotation.from_matrix(arrow_rotation).as_quat()
        quat_wxyz = [float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])]
        translate_attr = arrow_root.GetAttribute("xformOp:translate")
        if not translate_attr.IsValid():
            translate_attr = arrow_root.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3)
        translate_attr.Set(Gf.Vec3d(*[float(v) for v in actual_end_effector_pose[:3, 3]]))
        orient_attr = arrow_root.GetAttribute("xformOp:orient")
        if not orient_attr.IsValid():
            orient_attr = arrow_root.CreateAttribute("xformOp:orient", Sdf.ValueTypeNames.Quatf)
        orient_attr.Set(Gf.Quatf(quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]))
        order_attr = arrow_root.GetAttribute("xformOpOrder")
        order_attr.Set(["xformOp:translate", "xformOp:orient"])

        shaft = UsdGeom.Cylinder.Define(stage, f"{arrow_root_path}/Shaft")
        shaft_prim = shaft.GetPrim()
        shaft.GetRadiusAttr().Set(0.0028)
        shaft.GetHeightAttr().Set(0.03)
        shaft_display = shaft_prim.GetAttribute("primvars:displayColor")
        if not shaft_display.IsValid():
            shaft_display = shaft_prim.CreateAttribute("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray)
        shaft_display.Set([Gf.Vec3f(0.1, 0.5, 1.0)])
        shaft_translate = shaft_prim.GetAttribute("xformOp:translate")
        if not shaft_translate.IsValid():
            shaft_translate = shaft_prim.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3)
        shaft_translate.Set(Gf.Vec3d(0.0, 0.0, 0.015))
        shaft_order = shaft_prim.GetAttribute("xformOpOrder")
        shaft_order.Set(["xformOp:translate"])

        tip = UsdGeom.Cone.Define(stage, f"{arrow_root_path}/Tip")
        tip_prim = tip.GetPrim()
        tip.GetRadiusAttr().Set(0.005)
        tip.GetHeightAttr().Set(0.014)
        tip_display = tip_prim.GetAttribute("primvars:displayColor")
        if not tip_display.IsValid():
            tip_display = tip_prim.CreateAttribute("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray)
        tip_display.Set([Gf.Vec3f(0.1, 0.5, 1.0)])
        tip_translate = tip_prim.GetAttribute("xformOp:translate")
        if not tip_translate.IsValid():
            tip_translate = tip_prim.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3)
        tip_translate.Set(Gf.Vec3d(0.0, 0.0, 0.032))
        tip_order = tip_prim.GetAttribute("xformOpOrder")
        tip_order.Set(["xformOp:translate"])
