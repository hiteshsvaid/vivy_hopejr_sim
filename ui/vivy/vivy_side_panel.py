#!/usr/bin/env python3

from __future__ import annotations

from typing import Any


class VivySidePanel:
    _TEXT_NEUTRAL = 0xFFB8B8B8

    def __init__(self, *, width: int = 460, height: int = 380):
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
            with ui.ScrollingFrame():
                with ui.VStack(spacing=6, height=0):
                    section_style = {"color": 0xFFD8D8D8, "font_size": 13}
                    value_style = {"color": self._TEXT_NEUTRAL, "font_size": 13}
                    joint_value_style = {"color": self._TEXT_NEUTRAL, "font_size": 12}
                    header_style = {"color": 0xFFD8D8D8, "font_size": 12}
                    joint_name_style = {"color": 0xFFD8D8D8, "font_size": 12}

                    ui.Label("QUEST", style=section_style)
                    with ui.VGrid(column_count=2, column_widths=[110, 0], row_height=22):
                        ui.Label("position", style=value_style)
                        self._labels["quest_pos"] = ui.Label("-", style=value_style)
                        ui.Label("grip / trigger", style=value_style)
                        self._labels["quest_grip_trigger"] = ui.Label("-", style=value_style)

                    ui.Label("STATE", style=section_style)
                    with ui.VGrid(column_count=2, column_widths=[110, 0], row_height=22):
                        ui.Label("enabled / clutch", style=value_style)
                        self._labels["state_enabled_clutch"] = ui.Label("-", style=value_style)
                        ui.Label("anchor / frozen", style=value_style)
                        self._labels["state_anchor_frozen"] = ui.Label("-", style=value_style)

                    ui.Label("DELTA", style=section_style)
                    with ui.VGrid(column_count=2, column_widths=[110, 0], row_height=22):
                        ui.Label("quest", style=value_style)
                        self._labels["delta_quest"] = ui.Label("-", style=value_style)
                        ui.Label("world", style=value_style)
                        self._labels["delta_world"] = ui.Label("-", style=value_style)

                    ui.Spacer(height=4)
                    ui.Label("JOINTS", style=section_style)
                    with ui.VGrid(column_count=7, column_widths=[45, 45, 55, 65, 45, 45, 20], row_height=20):
                        ui.Label("MIN", style=header_style)
                        ui.Label("D_DEG", style=header_style)
                        ui.Label("CMD_DEG", style=header_style)
                        ui.Label("SERVO_CMD", style=header_style)
                        ui.Label("MAX", style=header_style)
                        ui.Label("MODE", style=header_style)
                        ui.Label("F", style=header_style)
                    for index in range(7):
                        with ui.VStack(spacing=2):
                            self._labels[f"joint_name_{index}"] = ui.Label("", style=joint_name_style)
                            with ui.VGrid(column_count=7, column_widths=[45, 45, 55, 65, 45, 45, 20], row_height=20):
                                self._labels[f"joint_min_deg_{index}"] = ui.Label("", style=joint_value_style)
                                self._labels[f"joint_delta_deg_{index}"] = ui.Label("", style=joint_value_style)
                                self._labels[f"joint_target_deg_{index}"] = ui.Label("", style=joint_value_style)
                                self._labels[f"joint_raw_cmd_{index}"] = ui.Label("", style=joint_value_style)
                                self._labels[f"joint_max_deg_{index}"] = ui.Label("", style=joint_value_style)
                                self._labels[f"joint_mode_{index}"] = ui.Label("", style=joint_value_style)
                                self._labels[f"joint_flag_{index}"] = ui.Label("", style=joint_value_style)
        self._dock_window(ui)

    @staticmethod
    def _fmt_vec3(value: Any) -> str:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            return "-"
        try:
            return f"({float(value[0]):+.3f}, {float(value[1]):+.3f}, {float(value[2]):+.3f})"
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
        anchor = "captured" if payload.get("quest_anchor_position") is not None else "pending"
        frozen = "yes" if payload.get("freeze_active") else "no"

        self._labels["quest_pos"].text = self._fmt_vec3(hand.get("position"))
        self._labels["quest_grip_trigger"].text = f"grip={grip:.2f}  trigger={trigger:.2f}"
        self._labels["state_enabled_clutch"].text = f"enabled={enabled}  clutch={clutch}"
        self._labels["state_anchor_frozen"].text = f"anchor={anchor}  frozen={frozen}"
        self._labels["delta_quest"].text = self._fmt_vec3(payload.get("quest_delta"))
        self._labels["delta_world"].text = self._fmt_vec3(payload.get("position_delta_world"))

        rows = payload.get("joint_display_rows") or []
        for index in range(7):
            self._labels[f"joint_name_{index}"].text = ""
            self._labels[f"joint_min_deg_{index}"].text = ""
            self._labels[f"joint_delta_deg_{index}"].text = ""
            self._labels[f"joint_target_deg_{index}"].text = ""
            self._labels[f"joint_raw_cmd_{index}"].text = ""
            self._labels[f"joint_max_deg_{index}"].text = ""
            self._labels[f"joint_mode_{index}"].text = ""
            self._labels[f"joint_flag_{index}"].text = ""
            if index < len(rows):
                row = rows[index]
                self._labels[f"joint_name_{index}"].text = str(row.get("joint") or "")
                self._labels[f"joint_min_deg_{index}"].text = str(row.get("min_deg") or "").strip()
                self._labels[f"joint_delta_deg_{index}"].text = str(row.get("delta_deg") or "").strip()
                self._labels[f"joint_target_deg_{index}"].text = str(row.get("target_deg") or "").strip()
                self._labels[f"joint_raw_cmd_{index}"].text = str(row.get("raw_cmd") or "").strip()
                self._labels[f"joint_max_deg_{index}"].text = str(row.get("max_deg") or "").strip()
                self._labels[f"joint_mode_{index}"].text = str(row.get("mode") or "")
                self._labels[f"joint_flag_{index}"].text = str(row.get("freeze") or "")
