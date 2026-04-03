#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


FLOW_CONTROL_PATH = Path("/tmp/vivy_flow_control.json")
VIVY_CONFIG_PATH = Path("/home/viaan/huggingface/lerobot/src/lerobot/robots/vivy/vivy_global_config.json")
RECORDINGS_DIR = Path("/tmp/hope_jr_quest_recordings")


class VivyFlowDetailPanel:
    _TEXT_HEADER = 0xFFE6EDF3
    _TEXT_NEUTRAL = 0xFFB8B8B8

    def __init__(self, *, width: int = 420, height: int = 180):
        self.width = width
        self.height = height
        self._window = None
        self._labels: dict[str, Any] = {}
        self._docked = False
        self._last_selected = None

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

    @staticmethod
    def _read_vivy_config() -> dict[str, Any]:
        try:
            return json.loads(VIVY_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}

    @staticmethod
    def _write_vivy_config(config: dict[str, Any]) -> None:
        VIVY_CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _normalize_axes(value: str) -> str:
        cleaned = str(value).strip().lower()
        if len(cleaned) != 3 or set(cleaned) != {"x", "y", "z"}:
            raise ValueError("axes must be a permutation of xyz, e.g. xyz or zxy")
        return cleaned

    def _load_axis_remap_fields_from_config(self) -> None:
        config = self._read_vivy_config()
        script_defaults = dict(config.get("script_editor_test_defaults") or {})
        controller_defaults = dict(config.get("controller_defaults") or {})
        axes = str(script_defaults.get("quest_position_axes", controller_defaults.get("quest_position_axes", "xyz")))
        signs = list(script_defaults.get("quest_position_signs", controller_defaults.get("quest_position_signs", [1.0, 1.0, 1.0])))
        try:
            self._labels["axis_axes_model"].set_value(axes)
            self._labels["axis_sign_x_model"].set_value(str(signs[0]))
            self._labels["axis_sign_y_model"].set_value(str(signs[1]))
            self._labels["axis_sign_z_model"].set_value(str(signs[2]))
            self._labels["axis_status"].text = "Edit and save. Restart the teleop run to apply."
        except Exception:
            pass

    def _save_axis_remap(self) -> None:
        try:
            axes = self._normalize_axes(self._labels["axis_axes_model"].get_value_as_string())
            signs = [
                float(self._labels["axis_sign_x_model"].get_value_as_string()),
                float(self._labels["axis_sign_y_model"].get_value_as_string()),
                float(self._labels["axis_sign_z_model"].get_value_as_string()),
            ]
            config = self._read_vivy_config()
            controller_defaults = dict(config.get("controller_defaults") or {})
            script_defaults = dict(config.get("script_editor_test_defaults") or {})
            controller_defaults["quest_position_axes"] = axes
            controller_defaults["quest_position_signs"] = list(signs)
            script_defaults["quest_position_axes"] = axes
            script_defaults["quest_position_signs"] = list(signs)
            config["controller_defaults"] = controller_defaults
            config["script_editor_test_defaults"] = script_defaults
            self._write_vivy_config(config)
            self._labels["axis_status"].text = f"Saved axes={axes} signs={signs}. Restart the teleop run to apply."
        except Exception as exc:
            self._labels["axis_status"].text = f"Save failed: {exc}"

    def _toggle_sim_view(self) -> None:
        control = self._read_flow_control()
        control["sim_view_enabled"] = not bool(control.get("sim_view_enabled", True))
        self._write_flow_control(control)

    def _save_replay_name(self) -> None:
        try:
            replay_name = str(self._labels["quest_replay_model"].get_value_as_string()).strip()
            if not replay_name:
                raise ValueError("replay name must not be empty")
            replay_path = RECORDINGS_DIR / f"{replay_name}.ndjson"
            if not replay_path.exists():
                raise ValueError(f"recording not found: {replay_name}")
            control = self._read_flow_control()
            control["replay_name"] = replay_name
            self._write_flow_control(control)
            self._labels["quest_status"].text = f"Saved replay={replay_name}. Restart the teleop run to apply."
        except Exception as exc:
            self._labels["quest_status"].text = f"Save failed: {exc}"

    def _ensure_window(self) -> None:
        if self._window is not None:
            return
        try:
            import omni.ui as ui
        except ImportError:
            return

        self._window = ui.Window("Vivy Flow Details", width=self.width, height=self.height)
        with self._window.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=6, height=0):
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
                    with ui.VStack(spacing=4) as quest_editor:
                        self._labels["quest_editor"] = quest_editor
                        ui.Label("Quest Replay", style=header_style)
                        with ui.HStack(height=26, spacing=6):
                            ui.Label("file", width=48, style=value_style)
                            replay_field = ui.StringField(width=220)
                            self._labels["quest_replay_model"] = replay_field.model
                        self._labels["quest_save_button"] = ui.Button(
                            "Save Replay Name",
                            height=28,
                            clicked_fn=lambda: self._save_replay_name(),
                        )
                        self._labels["quest_status"] = ui.Label("", style=value_style, word_wrap=True)
                    with ui.VStack(spacing=4) as axis_editor:
                        self._labels["axis_editor"] = axis_editor
                        ui.Label("Axis / Sign Remap", style=header_style)
                        with ui.HStack(height=26, spacing=6):
                            ui.Label("axes", width=48, style=value_style)
                            axis_field = ui.StringField(width=80)
                            self._labels["axis_axes_model"] = axis_field.model
                        with ui.HStack(height=26, spacing=6):
                            ui.Label("sign x", width=48, style=value_style)
                            sign_x = ui.StringField(width=60)
                            self._labels["axis_sign_x_model"] = sign_x.model
                            ui.Label("sign y", width=48, style=value_style)
                            sign_y = ui.StringField(width=60)
                            self._labels["axis_sign_y_model"] = sign_y.model
                            ui.Label("sign z", width=48, style=value_style)
                            sign_z = ui.StringField(width=60)
                            self._labels["axis_sign_z_model"] = sign_z.model
                        self._labels["axis_save_button"] = ui.Button(
                            "Save Axis Remap",
                            height=28,
                            clicked_fn=lambda: self._save_axis_remap(),
                        )
                        self._labels["axis_status"] = ui.Label("", style=value_style, word_wrap=True)
                    try:
                        quest_editor.visible = False
                        axis_editor.visible = False
                    except Exception:
                        pass
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
        if selected != self._last_selected:
            if selected == "axis_remap":
                self._load_axis_remap_fields_from_config()
            self._last_selected = selected

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
                self._labels["quest_editor"].visible = False
                self._labels["axis_editor"].visible = False
            except Exception:
                pass
        else:
            detail = {
                "quest": f"quest_mode={quest_mode} replay={replay_name}",
                "processor": "Processes Quest packets into teleop state",
                "teleop_state": f"waiting_for_anchor={waiting_for_anchor} freeze_active={freeze_active}",
                "ik": "Quest IK teleoperator branch",
                "quest_anchor_capture": "Place the Quest controller where you want, press A, and that current controller position/orientation plus the current robot pose become the new anchor.",
                "anchor_delta": "Computes current hand position minus anchor position.",
                "deadband": "Suppresses small motion below the configured threshold.",
                "axis_remap": "Applies axis reorder and sign flips to Quest deltas.",
                "world_transform": "Applies scale and world-frame rotation.",
                "target_pose": "Builds the target pose used for downstream consumers.",
                "fanout": "Consumes teleop state for real/log sinks",
                "real": "Hardware sink branch",
                "log": f"log sink branch output={robot_output}",
            }.get(selected, "Select a node from the flow tree.")
            self._labels["detail"].text = detail
            self._labels["action_hint"].text = ""
            try:
                self._labels["sim_toggle_button"].visible = False
                self._labels["quest_editor"].visible = selected == "quest" and quest_mode == "replay"
                self._labels["axis_editor"].visible = selected == "axis_remap"
            except Exception:
                pass
            if selected == "quest" and quest_mode == "replay":
                try:
                    self._labels["quest_replay_model"].set_value(replay_name if replay_name != "-" else "")
                    self._labels["quest_status"].text = "Save a different replay name, then restart the teleop run."
                except Exception:
                    pass
