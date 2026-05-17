#!/usr/bin/env python3

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


FLOW_CONTROL_PATH = Path("/tmp/vivy_flow_control.json")
ISAAC_SIM_EXTSCACHE_DIR = Path("/isaac-sim/extscache")


def _resolve_extscache_path(pattern: str, relative_path: str) -> Path:
    matches = sorted(ISAAC_SIM_EXTSCACHE_DIR.glob(pattern))
    if not matches:
        return ISAAC_SIM_EXTSCACHE_DIR / relative_path
    return matches[-1] / relative_path


KIT_INPUT_ICON_PATH = _resolve_extscache_path(
    "omni.kit.window.material_graph-*", "icons/InputNode.svg"
)
KIT_OUTPUT_ICON_PATH = _resolve_extscache_path(
    "omni.kit.window.material_graph-*", "icons/OutputNode.svg"
)
GRAPH_ICON_DIR = _resolve_extscache_path("omni.graph.window.core-*", "icons/node")
CONTENT_ICON_DIR = _resolve_extscache_path(
    "omni.kit.window.content_browser-*", "icons/NvidiaDark"
)


class VivyFlowPanel:
    _TEXT_HEADER = 0xFFE6EDF3
    _TEXT_NEUTRAL = 0xFFB8B8B8
    _TEXT_ACTIVE = 0xFF7EE787
    _TEXT_INACTIVE = 0xFF6E7681
    _TEXT_WARN = 0xFFFFC857
    _TEXT_SELECTED = 0xFFFFFFFF

    _TREE = {
        "root": ["quest"],
        "quest": ["processor", "ik", "sim"],
        "processor": ["teleop_state"],
        "teleop_state": [],
        "ik": ["joint_targets", "target"],
        "joint_targets": [],
        "target": ["real", "sim_output"],
        "sim": ["sim_input", "sim_joint_targets"],
        "sim_input": [],
        "sim_joint_targets": [],
        "real": [],
        "sim_output": [],
    }

    _PARENTS = {
        "quest": "root",
        "processor": "quest",
        "teleop_state": "processor",
        "ik": "quest",
        "sim": "quest",
        "joint_targets": "ik",
        "target": "ik",
        "real": "target",
        "sim_output": "target",
        "sim_input": "sim",
        "sim_joint_targets": "sim",
    }

    _INDENT = {
        "root": 0,
        "quest": 1,
        "processor": 2,
        "teleop_state": 3,
        "ik": 2,
        "sim": 2,
        "joint_targets": 3,
        "target": 3,
        "real": 4,
        "sim_output": 4,
        "sim_input": 3,
        "sim_joint_targets": 3,
    }

    _DEFAULT_EXPANDED = {
        "processor": False,
        "sim": False,
    }

    _INPUT_NODES = {"sim_input", "sim_joint_targets"}
    _OUTPUT_NODES = {"teleop_state", "joint_targets"}
    _NODE_ICON_FILES = {
        "root": CONTENT_ICON_DIR / "usd_stage_256.png",
        "quest": GRAPH_ICON_DIR / "type_input_noBorder_dark.svg",
        "processor": GRAPH_ICON_DIR / "type_script_noBorder_dark.svg",
        "ik": GRAPH_ICON_DIR / "type_function_noBorder_dark.svg",
        "target": GRAPH_ICON_DIR / "type_io_noBorder_dark.svg",
        "real": GRAPH_ICON_DIR / "type_io_noBorder_dark.svg",
        "sim_output": GRAPH_ICON_DIR / "type_rendering_noBorder_dark.svg",
        "sim": GRAPH_ICON_DIR / "type_rendering_noBorder_dark.svg",
    }

    def __init__(self, *, width: int = 420, height: int = 320):
        self.width = width
        self.height = height
        self._window = None
        self._docked = False
        self._row_frames: dict[str, Any] = {}
        self._toggle_buttons: dict[str, Any] = {}
        self._label_widgets: dict[str, Any] = {}
        self._select_buttons: dict[str, Any] = {}
        self._icon_widgets: dict[str, Any] = {}

    def _dock_window(self, ui_module: Any) -> None:
        if self._window is None or self._docked:
            return
        try:
            workspace = ui_module.Workspace
            right_target = workspace.get_window("Vivy Quest") or workspace.get_window("Vivy Side") or workspace.get_window("Stage")
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

    def _toggle_node(self, node_name: str) -> None:
        control = self._read_flow_control()
        expanded = dict(control.get("expanded") or {})
        default_value = bool(self._DEFAULT_EXPANDED.get(node_name, True))
        expanded[node_name] = not bool(expanded.get(node_name, default_value))
        control["expanded"] = expanded
        self._write_flow_control(control)

    def _is_expanded(self, node_name: str, flow_state: dict[str, Any]) -> bool:
        expanded = dict(flow_state.get("expanded") or {})
        return bool(expanded.get(node_name, self._DEFAULT_EXPANDED.get(node_name, True)))

    def _is_visible(self, node_name: str, flow_state: dict[str, Any]) -> bool:
        parent = self._PARENTS.get(node_name)
        while parent is not None:
            if not self._is_expanded(parent, flow_state):
                return False
            parent = self._PARENTS.get(parent)
        return True

    def _leaf_icon_path(self, key: str) -> str | None:
        if key in self._INPUT_NODES and KIT_INPUT_ICON_PATH.exists():
            return str(KIT_INPUT_ICON_PATH)
        if key in self._OUTPUT_NODES and KIT_OUTPUT_ICON_PATH.exists():
            return str(KIT_OUTPUT_ICON_PATH)
        return None

    def _leaf_icon_text(self, key: str) -> str:
        if key in self._INPUT_NODES:
            return "<"
        if key in self._OUTPUT_NODES:
            return ">"
        return "o"

    def _node_icon_path(self, key: str) -> str | None:
        if key in self._INPUT_NODES or key in self._OUTPUT_NODES:
            return None
        path = self._NODE_ICON_FILES.get(key)
        if path is not None and path.exists():
            return str(path)
        return None

    def _row(self, ui: Any, key: str) -> None:
        indent = self._INDENT[key]
        with ui.HStack(height=22, spacing=4) as row:
            self._row_frames[key] = row
            ui.Spacer(width=indent * 16)
            if self._TREE.get(key):
                self._toggle_buttons[key] = ui.Button(
                    "v",
                    width=18,
                    clicked_fn=lambda k=key: self._toggle_node(k),
                    style={
                        "background_color": 0x00000000,
                        "border_width": 0,
                        "color": self._TEXT_NEUTRAL,
                        "font_size": 13,
                    },
                )
            else:
                icon_path = self._leaf_icon_path(key)
                if icon_path is not None:
                    self._toggle_buttons[key] = ui.Image(icon_path, width=16, height=16)
                else:
                    self._toggle_buttons[key] = ui.Label(
                        self._leaf_icon_text(key),
                        width=18,
                        style={"color": self._TEXT_INACTIVE, "font_size": 12},
                    )
            node_icon_path = self._node_icon_path(key)
            if node_icon_path is not None:
                self._icon_widgets[key] = ui.Image(node_icon_path, width=16, height=16)
            else:
                self._icon_widgets[key] = ui.Spacer(width=16)
            self._select_buttons[key] = ui.Button(
                "o",
                width=18,
                clicked_fn=lambda k=key: self._select_node(k),
                style={
                    "background_color": 0x00000000,
                    "border_width": 0,
                    "color": self._TEXT_NEUTRAL,
                    "font_size": 13,
                },
            )
            self._label_widgets[key] = ui.Label(
                "-",
                width=max(160, self.width - indent * 16 - 82),
                alignment=ui.Alignment.LEFT_CENTER,
                name="TreeView.Item",
                style={"color": self._TEXT_NEUTRAL, "font_size": 13},
            )
            try:
                self._label_widgets[key].set_mouse_pressed_fn(lambda x, y, b, m, k=key: self._select_node(k))
            except Exception:
                pass

    def _ensure_window(self) -> None:
        if self._window is not None:
            return
        try:
            import omni.ui as ui
        except ImportError:
            return

        self._window = ui.Window("Vivy Flow", width=self.width, height=self.height)
        try:
            self._window.deferred_dock_in("Vivy Quest")
        except Exception:
            try:
                self._window.deferred_dock_in("Stage")
            except Exception:
                pass
        with self._window.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=6, height=0):
                    header_style = {"color": self._TEXT_HEADER, "font_size": 15}
                    ui.Label("Vivy Teleop", style=header_style)
                    for key in [
                        "root",
                        "quest",
                        "processor",
                        "teleop_state",
                        "ik",
                        "joint_targets",
                        "target",
                        "real",
                        "sim_output",
                        "sim",
                        "sim_input",
                        "sim_joint_targets",
                    ]:
                        self._row(ui, key)
        self._dock_window(ui)

    def _set_row(self, key: str, text: str, state: str, selected: bool, flow_state: dict[str, Any]) -> None:
        color = {
            "active": self._TEXT_ACTIVE,
            "inactive": self._TEXT_INACTIVE,
            "warn": self._TEXT_WARN,
            "neutral": self._TEXT_NEUTRAL,
        }.get(state, self._TEXT_NEUTRAL)
        if selected:
            color = self._TEXT_SELECTED
        try:
            self._label_widgets[key].text = text
            self._label_widgets[key].style = {"color": color, "font_size": 13}
            self._select_buttons[key].text = "*" if selected else "o"
            self._select_buttons[key].style = {
                "background_color": 0x00000000,
                "border_width": 0,
                "color": color,
                "font_size": 13,
            }
            toggle = self._toggle_buttons[key]
            if key in self._TREE and self._TREE[key]:
                toggle.text = "v" if self._is_expanded(key, flow_state) else ">"
                toggle.style = {
                    "background_color": 0x00000000,
                    "border_width": 0,
                    "color": self._TEXT_NEUTRAL,
                    "font_size": 13,
                }
            self._row_frames[key].visible = self._is_visible(key, flow_state) if key != "root" else True
        except Exception:
            pass

    def update(self, payload: dict[str, Any] | None = None, flow_state: dict[str, Any] | None = None) -> None:
        self._ensure_window()
        if not self._label_widgets:
            return
        try:
            import omni.ui as ui
            self._dock_window(ui)
        except Exception:
            pass

        payload = payload or {}
        flow_state = flow_state or {}
        hand = payload.get("hand_state") or {}

        selected = str(flow_state.get("selected_node") or "sim")
        if selected not in self._label_widgets:
            selected = "sim"
        quest_mode = str(flow_state.get("quest_mode") or "?")
        target = str(flow_state.get("target") or "?")
        robot_output = str(flow_state.get("robot_output") or "-")
        freeze_active = bool(payload.get("freeze_active", False))
        source_active = bool(hand)
        state_active = payload.get("target_pose_model") is not None
        sim_view_enabled = bool(flow_state.get("sim_view_enabled", True))
        sim_branch_active = sim_view_enabled
        robot_branch_active = target == "arm" and robot_output == "arm"
        sim_output_branch_active = target == "sim" and robot_output == "sim"

        self._set_row("root", "Vivy Teleop", "neutral", selected == "root", flow_state)
        self._set_row(
            "quest",
            f"Quest  live={'ON' if quest_mode == 'live' else 'OFF'}  replay={'ON' if quest_mode == 'replay' else 'OFF'}",
            "active" if source_active else "inactive",
            selected == "quest",
            flow_state,
        )
        self._set_row("processor", "quest_signal_processor", "active" if source_active else "inactive", selected == "processor", flow_state)
        self._set_row("teleop_state", "Out: Teleop State", "active" if state_active or source_active else "inactive", selected == "teleop_state", flow_state)
        self._set_row("ik", "IK  (quest_ik_arm)", "warn" if freeze_active else ("active" if target in {"arm", "sim"} else "inactive"), selected == "ik", flow_state)
        self._set_row("sim", f"Sim View  {'ON' if sim_view_enabled else 'OFF'}", "active" if sim_branch_active and sim_view_enabled else "inactive", selected == "sim", flow_state)
        self._set_row("joint_targets", "Out: Joint Targets", "warn" if freeze_active else ("active" if target in {"arm", "sim"} else "inactive"), selected == "joint_targets", flow_state)
        self._set_row("target", "vivy_target_sink", "warn" if freeze_active else ("active" if target in {"arm", "sim"} else "inactive"), selected == "target", flow_state)
        self._set_row("real", "Real Arm", "warn" if freeze_active else ("active" if robot_branch_active else "inactive"), selected == "real", flow_state)
        self._set_row("sim_output", "Sim Output", "active" if sim_output_branch_active else "inactive", selected == "sim_output", flow_state)
        self._set_row("sim_input", "In: Teleop State", "active" if state_active or source_active else "inactive", selected == "sim_input", flow_state)
        self._set_row("sim_joint_targets", "In: Joint Targets", "warn" if freeze_active else ("active" if target in {"arm", "sim"} else "inactive"), selected == "sim_joint_targets", flow_state)
