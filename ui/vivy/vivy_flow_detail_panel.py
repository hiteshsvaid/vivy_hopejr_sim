#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


FLOW_CONTROL_PATH = Path("/tmp/vivy_flow_control.json")


class VivyFlowDetailPanel:
    _TEXT_HEADER = 0xFFE6EDF3
    _TEXT_NEUTRAL = 0xFFB8B8B8

    def __init__(self, *, width: int = 420, height: int = 180):
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
                ui_module.Workspace.show_window("Vivy Flow Details", True)
                self._window.dock_in(bottom_target, ui_module.DockPosition.SAME)
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

    def _toggle_sim_view(self) -> None:
        control = self._read_flow_control()
        control["sim_view_enabled"] = not bool(control.get("sim_view_enabled", True))
        self._write_flow_control(control)

    def _ensure_window(self) -> None:
        if self._window is not None:
            return
        try:
            import omni.ui as ui
        except ImportError:
            return

        self._window = ui.Window("Vivy Flow Details", width=self.width, height=self.height)
        with self._window.frame:
            with ui.VStack(spacing=8):
                header_style = {"color": self._TEXT_HEADER, "font_size": 15}
                value_style = {"color": self._TEXT_NEUTRAL, "font_size": 13}
                ui.Label("Selected Node", style=header_style)
                self._labels["selected"] = ui.Label("-", style=value_style, word_wrap=True)
                self._labels["detail"] = ui.Label("-", style=value_style, word_wrap=True)
                self._labels["action_hint"] = ui.Label("", style=value_style, word_wrap=True)
                self._labels["sim_toggle_button"] = ui.Button(
                    "Toggle Sim View",
                    height=28,
                    clicked_fn=lambda: self._toggle_sim_view(),
                )
        self._dock_window(ui)

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
        selected = str(flow_state.get("selected_node") or "sim")
        self._labels["selected"].text = selected

        waiting_for_anchor = bool(payload.get("waiting_for_anchor", True))
        freeze_active = bool(payload.get("freeze_active", False))
        sim_view_enabled = bool(flow_state.get("sim_view_enabled", True))
        replay_name = str(flow_state.get("replay_name") or "-")
        quest_mode = str(flow_state.get("quest_mode") or "?")
        robot_output = str(flow_state.get("robot_output") or "-")

        if selected == "sim":
            self._labels["detail"].text = (
                f"Sim target marker branch\n"
                f"enabled={sim_view_enabled}"
            )
            self._labels["action_hint"].text = "Use the button below to toggle the sim branch."
            try:
                self._labels["sim_toggle_button"].visible = True
                self._labels["sim_toggle_button"].text = "Turn Sim View Off" if sim_view_enabled else "Turn Sim View On"
            except Exception:
                pass
        else:
            detail = {
                "quest": f"quest_mode={quest_mode} replay={replay_name}",
                "processor": "Processes Quest packets into teleop state",
                "teleop_state": f"waiting_for_anchor={waiting_for_anchor} freeze_active={freeze_active}",
                "ik": "Quest IK teleoperator branch",
                "fanout": "Consumes teleop state for real/log sinks",
                "real": "Hardware sink branch",
                "log": f"log sink branch output={robot_output}",
            }.get(selected, "Select a node from the flow tree.")
            self._labels["detail"].text = detail
            self._labels["action_hint"].text = ""
            try:
                self._labels["sim_toggle_button"].visible = False
            except Exception:
                pass
