#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


FLOW_CONTROL_PATH = Path("/tmp/vivy_flow_control.json")


class VivyFlowPanel:
    _TEXT_HEADER = 0xFFE6EDF3
    _TEXT_NEUTRAL = 0xFFB8B8B8
    _TEXT_ACTIVE = 0xFF7EE787
    _TEXT_INACTIVE = 0xFF6E7681
    _TEXT_WARN = 0xFFFFC857

    def __init__(self, *, width: int = 420, height: int = 320):
        self.width = width
        self.height = height
        self._window = None
        self._labels: dict[str, Any] = {}
        self._row_buttons: dict[str, Any] = {}
        self._docked = False

    def _dock_window(self, ui_module: Any) -> None:
        if self._window is None or self._docked:
            return
        try:
            workspace = ui_module.Workspace
            right_target = workspace.get_window("Vivy Quest") or workspace.get_window("Hope Jr Side") or workspace.get_window("Stage")
            if right_target is not None:
                ui_module.Workspace.show_window("Vivy Flow", True)
                self._window.dock_in(right_target, ui_module.DockPosition.SAME)
                try:
                    self._window.focus()
                except Exception:
                    pass
                self._docked = True
        except Exception:
            pass

    @staticmethod
    def _read_flow_control() -> dict:
        if not FLOW_CONTROL_PATH.exists():
            return {}
        try:
            return json.loads(FLOW_CONTROL_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}

    @staticmethod
    def _write_flow_control(control: dict[str, Any]) -> None:
        FLOW_CONTROL_PATH.write_text(json.dumps(control) + "\n", encoding="utf-8")

    def _select_node(self, node_name: str) -> None:
        control = self._read_flow_control()
        control["selected_node"] = str(node_name)
        self._write_flow_control(control)

    def _row(self, ui: Any, key: str, indent: int = 0, selectable: bool = False) -> None:
        with ui.HStack(height=22, spacing=6):
            ui.Spacer(width=indent * 16)
            if selectable:
                self._row_buttons[key] = ui.Button(
                    "-",
                    width=0,
                    clicked_fn=lambda: self._select_node(key),
                    style={
                        "alignment": 1,
                        "background_color": 0x00000000,
                        "border_width": 0,
                        "color": self._TEXT_NEUTRAL,
                        "font_size": 13,
                    },
                )
            else:
                ui.Spacer(width=18)
                self._row_buttons[key] = None
            self._labels[key] = ui.Label("-", word_wrap=False, style={"color": self._TEXT_NEUTRAL, "font_size": 13})

    def _ensure_window(self) -> None:
        if self._window is not None:
            return
        try:
            import omni.ui as ui
        except ImportError:
            return

        self._window = ui.Window("Vivy Flow", width=self.width, height=self.height)
        with self._window.frame:
            with ui.VStack(spacing=8):
                header_style = {"color": self._TEXT_HEADER, "font_size": 15}

                ui.Label("Vivy Teleop", style=header_style)
                self._row(ui, "root", indent=0)
                self._row(ui, "quest", indent=1, selectable=True)
                self._row(ui, "processor", indent=2, selectable=True)
                self._row(ui, "teleop_state", indent=3, selectable=True)
                self._row(ui, "ik", indent=4, selectable=True)
                self._row(ui, "fanout", indent=5, selectable=True)
                self._row(ui, "real", indent=6, selectable=True)
                self._row(ui, "log", indent=6, selectable=True)
                self._row(ui, "sim", indent=4, selectable=True)
        self._dock_window(ui)

    def _set_row(self, key: str, text: str, state: str = "neutral") -> None:
        color = {
            "active": self._TEXT_ACTIVE,
            "inactive": self._TEXT_INACTIVE,
            "warn": self._TEXT_WARN,
            "neutral": self._TEXT_NEUTRAL,
        }.get(state, self._TEXT_NEUTRAL)
        try:
            button = self._row_buttons.get(key)
            if button is not None:
                button.text = text
                button.style = {
                    "alignment": 1,
                    "background_color": 0x00000000,
                    "border_width": 0,
                    "color": color,
                    "font_size": 13,
                }
                self._labels[key].text = ""
            else:
                label = self._labels[key]
                label.text = text
                label.style = {"color": color, "font_size": 13}
        except Exception:
            pass

    def update(self, payload: dict[str, Any] | None = None, flow_state: dict[str, Any] | None = None) -> None:
        self._ensure_window()
        if not self._labels:
            return
        try:
            import omni.ui as ui
            self._dock_window(ui)
        except Exception:
            pass

        payload = payload or {}
        flow_state = flow_state or {}
        hand = payload.get("hand_state") or {}

        quest_mode = str(flow_state.get("quest_mode") or "?")
        replay_name = str(flow_state.get("replay_name") or "-")
        target = str(flow_state.get("target") or "?")
        robot_output = str(flow_state.get("robot_output") or "-")
        waiting_for_anchor = bool(payload.get("waiting_for_anchor", True))
        freeze_active = bool(payload.get("freeze_active", False))
        freeze_joint = payload.get("freeze_joint_name") or "unknown"
        source_active = bool(hand)
        state_active = payload.get("target_pose_model") is not None
        sim_view_enabled = bool(flow_state.get("sim_view_enabled", True))
        sim_branch_active = bool(flow_state.get("sim_target_view", False))
        robot_branch_active = target == "real" and robot_output == "arm"
        log_branch_active = target == "real" and robot_output == "log"

        self._set_row("root", "Vivy Teleop", "neutral")
        self._set_row(
            "quest",
            f"+- Quest  live={'ON' if quest_mode == 'live' else 'OFF'}  replay={'ON' if quest_mode == 'replay' else 'OFF'}",
            "active" if source_active else "inactive",
        )
        self._set_row("processor", "+- quest_signal_processor", "active" if source_active else "inactive")
        self._set_row("teleop_state", "+- Teleop State", "active" if state_active or source_active else "inactive")
        self._set_row("ik", "+- IK  (quest_ik_arm)", "warn" if freeze_active else ("active" if target == "real" else "inactive"))
        self._set_row("fanout", "+- fanout_target_arm", "warn" if freeze_active else ("active" if target == 'real' else "inactive"))
        self._set_row("real", "+- Real Arm", "warn" if freeze_active else ("active" if robot_branch_active else "inactive"))
        self._set_row("log", "+- Log Sink", "active" if log_branch_active else "inactive")
        self._set_row("sim", f"+- Sim View  {'ON' if sim_view_enabled else 'OFF'}", "active" if sim_branch_active and sim_view_enabled else "inactive")
