#!/usr/bin/env python3

from typing import Any


class HopeJrControlProfilePanel:
    _TEXT_NEUTRAL = 0xFFB8B8B8

    def __init__(self, *, width: int = 680, height: int = 88):
        self.width = width
        self.height = height
        self._window = None
        self._labels: dict[str, Any] = {}
        self._docked = False

    def _dock_window(self, ui_module: Any) -> None:
        if self._window is None or self._docked:
            return
        try:
            workspace = ui_module.Workspace
            bottom_target = (
                workspace.get_window("Console")
                or workspace.get_window("Script Editor")
                or workspace.get_window("Property")
                or workspace.get_window("Stage")
            )
            if bottom_target is not None:
                ui_module.Workspace.show_window("Hope Jr Bottom", True)
                self._window.dock_in(bottom_target, ui_module.DockPosition.SAME)
                self._docked = True
        except Exception:
            pass

    def _ensure_window(self) -> None:
        if self._window is not None:
            return
        try:
            import omni.ui as ui
        except ImportError:
            return

        self._window = ui.Window("Hope Jr Bottom", width=self.width, height=self.height)
        with self._window.frame:
            with ui.VStack(spacing=4):
                header_style = {"color": 0xFFD8D8D8, "font_size": 13}
                value_style = {"color": self._TEXT_NEUTRAL, "font_size": 13}
                with ui.VGrid(column_count=3, column_widths=[130, 0, 0], spacing=8):
                    ui.Label("PROFILE", style=header_style)
                    self._labels["profile"] = ui.Label("-", style=value_style, word_wrap=True)
                    self._labels["summary"] = ui.Label("-", style=value_style, word_wrap=True)
                with ui.VGrid(column_count=2, column_widths=[130, 0], spacing=8):
                    ui.Label("HELD JOINTS", style=header_style)
                    self._labels["held"] = ui.Label("-", style=value_style, word_wrap=True)
                with ui.VGrid(column_count=2, column_widths=[130, 0], spacing=8):
                    ui.Label("SOLVE JOINTS", style=header_style)
                    self._labels["solve"] = ui.Label("-", style=value_style, word_wrap=True)
        self._dock_window(ui)

    def update(self, controller, debug: dict[str, Any] | None = None) -> None:
        self._ensure_window()
        if not self._labels:
            return
        try:
            import omni.ui as ui
            self._dock_window(ui)
        except Exception:
            pass
        debug = debug or controller.last_debug_payload or {}
        result = debug.get("result") or {}
        joint_names = list(result.get("joint_names") or [])
        joint_control_profile = str(result.get("joint_control_profile") or getattr(controller, "position_only_joint_control_profile", "all_solve_v1"))
        joint_control_modes = list(result.get("joint_control_modes") or getattr(controller, "position_only_joint_control_modes", []))
        held_joint_names = [name for name, mode in zip(joint_names, joint_control_modes) if mode != "solve"]
        solve_joint_names = [name for name, mode in zip(joint_names, joint_control_modes) if mode == "solve"]
        self._labels["profile"].text = joint_control_profile
        self._labels["summary"].text = f"solve={len(solve_joint_names)} held={len(held_joint_names)}"
        self._labels["held"].text = ", ".join(held_joint_names) if held_joint_names else "none"
        self._labels["solve"].text = ", ".join(solve_joint_names) if solve_joint_names else "none"
