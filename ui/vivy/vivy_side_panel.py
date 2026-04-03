#!/usr/bin/env python3

from __future__ import annotations

from typing import Any


class VivySidePanel:
    _TEXT_NEUTRAL = 0xFFB8B8B8

    def __init__(self, *, width: int = 420, height: int = 240):
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
            right_target = workspace.get_window("Hope Jr Side") or workspace.get_window("Stage")
            if right_target is not None:
                ui_module.Workspace.show_window("Vivy Quest", True)
                self._window.dock_in(right_target, ui_module.DockPosition.SAME)
                try:
                    self._window.focus()
                except Exception:
                    pass
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

        self._window = ui.Window("Vivy Quest", width=self.width, height=self.height)
        try:
            self._window.focus()
        except Exception:
            pass
        with self._window.frame:
            with ui.VStack(spacing=4):
                section_style = {"color": 0xFFD8D8D8, "font_size": 13}
                value_style = {"color": self._TEXT_NEUTRAL, "font_size": 13}
                with ui.VGrid(column_count=2, column_widths=[120, 0], spacing=8):
                    ui.Label("STATE", style=section_style)
                    self._labels["state"] = ui.Label("-", style=value_style, word_wrap=True)
                    ui.Label("HAND POS", style=section_style)
                    self._labels["hand_pos"] = ui.Label("-", style=value_style, word_wrap=True)
                    ui.Label("QUEST DELTA", style=section_style)
                    self._labels["quest_delta"] = ui.Label("-", style=value_style, word_wrap=True)
                    ui.Label("WORLD DELTA", style=section_style)
                    self._labels["world_delta"] = ui.Label("-", style=value_style, word_wrap=True)
                    ui.Label("GRIP/TRIGGER", style=section_style)
                    self._labels["grip_trigger"] = ui.Label("-", style=value_style, word_wrap=True)
                    ui.Label("BUTTONS", style=section_style)
                    self._labels["buttons"] = ui.Label("-", style=value_style, word_wrap=True)
                    ui.Label("ANCHOR", style=section_style)
                    self._labels["anchor"] = ui.Label("-", style=value_style, word_wrap=True)
                    ui.Label("FREEZE", style=section_style)
                    self._labels["freeze"] = ui.Label("-", style=value_style, word_wrap=True)
        self._dock_window(ui)

    @staticmethod
    def _fmt_vec3(value: Any) -> str:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            return "-"
        try:
            return ", ".join(f"{float(v):+.3f}" for v in value)
        except Exception:
            return "-"

    def update(self, payload: dict[str, Any] | None = None) -> None:
        self._ensure_window()
        if not self._labels:
            return
        try:
            import omni.ui as ui
            self._dock_window(ui)
        except Exception:
            pass

        payload = payload or {}
        hand = payload.get("hand_state") or {}

        grip = float(hand.get("grip", 0.0)) if isinstance(hand, dict) else 0.0
        trigger = float(hand.get("trigger", 0.0)) if isinstance(hand, dict) else 0.0
        enabled = bool(hand.get("enabled", True)) if isinstance(hand, dict) else False
        clutch = bool(hand.get("clutch", False)) if isinstance(hand, dict) else False
        buttons = (
            f"A={int(bool(hand.get('a_pressed', False)))} "
            f"B={int(bool(hand.get('b_pressed', False)))} "
            f"X={int(bool(hand.get('x_pressed', False)))} "
            f"Y={int(bool(hand.get('y_pressed', False)))} "
            f"P={int(bool(hand.get('primary_button', False)))} "
            f"S={int(bool(hand.get('secondary_button', False)))}"
        )

        anchor = "captured" if payload.get("quest_anchor_position") is not None else "pending"
        freeze = "no"
        if payload.get("freeze_active"):
            freeze_joint = payload.get("freeze_joint_name") or "unknown"
            freeze = f"yes ({freeze_joint})"

        self._labels["state"].text = f"enabled={enabled} clutch={clutch}"
        self._labels["hand_pos"].text = self._fmt_vec3(hand.get("position"))
        self._labels["quest_delta"].text = self._fmt_vec3(payload.get("quest_delta"))
        self._labels["world_delta"].text = self._fmt_vec3(payload.get("position_delta_world"))
        self._labels["grip_trigger"].text = f"grip={grip:.2f} trigger={trigger:.2f}"
        self._labels["buttons"].text = buttons
        self._labels["anchor"].text = anchor
        self._labels["freeze"].text = freeze
