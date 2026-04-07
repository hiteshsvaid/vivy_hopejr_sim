#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_DETAIL_SECTION_DIR = Path(__file__).resolve().parent / "detail_sections"
QuestDetailSection = _load_module(
    "vivy_quest_detail_section", _DETAIL_SECTION_DIR / "quest_detail_section.py"
).QuestDetailSection
MappingDetailSection = _load_module(
    "vivy_mapping_detail_section", _DETAIL_SECTION_DIR / "mapping_detail_section.py"
).MappingDetailSection
IkDetailSection = _load_module(
    "vivy_ik_detail_section", _DETAIL_SECTION_DIR / "ik_detail_section.py"
).IkDetailSection

FLOW_CONTROL_PATH = Path("/tmp/vivy_flow_control.json")
VIVY_CONFIG_PATH = Path("/home/viaan/huggingface/lerobot/src/lerobot/robots/vivy/vivy_global_config.json")
RECORDINGS_DIR = Path("/tmp/hope_jr_quest_recordings")


class VivyFlowDetailPanel:
    _TEXT_HEADER = 0xFFE6EDF3
    _TEXT_NEUTRAL = 0xFFB8B8B8
    _INPUT_SOURCE_NODE = {
        "sim_input": "teleop_state",
        "sim_joint_targets": "joint_targets",
    }
    _INPUT_SOURCE_LABEL = {
        "sim_input": "Out: Teleop State",
        "sim_joint_targets": "Out: Joint Targets",
    }

    def __init__(self, *, width: int = 420, height: int = 180):
        self.width = width
        self.height = height
        self._window = None
        self._labels: dict[str, Any] = {}
        self._docked = False
        self._last_selected = None
        self._last_saved_replay_name: str | None = None
        self._last_saved_record_name: str | None = None
        self._quest_recording_names: list[str] = []
        self._quest_replay_container = None
        self._ik_joint_names: list[str] = []
        self._ik_jacobian_modes = ["finite_difference", "analytic"]
        self._ik_jacobian_mode_container = None
        self._joint_axis_options = ["X", "Y", "Z", "-X", "-Y", "-Z"]
        self._last_payload_rows: dict[str, dict[str, Any]] = {}
        self._last_ik_table_signature: tuple | None = None
        self._quest_section = QuestDetailSection(self)
        self._mapping_section = MappingDetailSection(self)
        self._ik_section = IkDetailSection(self)

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
            self._labels["target_max_delta_model"].set_value(str(controller_defaults.get("target_max_delta_m_per_tick", 0.0)))
            self._labels["axis_status"].text = "Edit and save. Applies live."
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
            target_max_delta = float(self._labels["target_max_delta_model"].get_value_as_string())
            controller_defaults["quest_position_axes"] = axes
            controller_defaults["quest_position_signs"] = list(signs)
            controller_defaults["target_max_delta_m_per_tick"] = target_max_delta
            script_defaults["quest_position_axes"] = axes
            script_defaults["quest_position_signs"] = list(signs)
            config["controller_defaults"] = controller_defaults
            config["script_editor_test_defaults"] = script_defaults
            self._write_vivy_config(config)
            self._labels["axis_status"].text = f"Saved axes={axes} signs={signs} target_max_delta={target_max_delta}. Applies live."
        except Exception as exc:
            self._labels["axis_status"].text = f"Save failed: {exc}"

    def _toggle_sim_view(self) -> None:
        control = self._read_flow_control()
        control["sim_view_enabled"] = not bool(control.get("sim_view_enabled", True))
        self._write_flow_control(control)

    def _toggle_pitch_frames(self) -> None:
        control = self._read_flow_control()
        control["show_pitch_frames"] = not bool(control.get("show_pitch_frames", False))
        self._write_flow_control(control)

    def _jump_to_source_output(self) -> None:
        selected = str(self._read_flow_control().get("selected_node") or "")
        target = self._INPUT_SOURCE_NODE.get(selected)
        if not target:
            return
        control = self._read_flow_control()
        control["selected_node"] = target
        self._write_flow_control(control)

    @staticmethod
    def _list_recording_names() -> list[str]:
        if not RECORDINGS_DIR.exists():
            return []
        return sorted(path.stem for path in RECORDINGS_DIR.glob("*.ndjson"))

    def _refresh_replay_dropdown(self, selected_name: str | None = None) -> None:
        try:
            import omni.ui as ui
        except ImportError:
            return
        container = self._quest_replay_container
        if container is None:
            return
        names = self._list_recording_names()
        self._quest_recording_names = names
        display_names = names or ["(none)"]
        try:
            container.clear()
        except Exception:
            pass
        with container:
            ui.Label("replay", width=48, style={"color": self._TEXT_NEUTRAL, "font_size": 13})
            combo = ui.ComboBox(0, *display_names, width=220)
            self._labels["quest_replay_combo"] = combo
            self._labels["quest_replay_combo_model"] = combo.model
        if names:
            if selected_name in names:
                selected_index = names.index(selected_name)
            else:
                selected_index = 0
            self._labels["quest_replay_combo_model"].get_item_value_model().set_value(selected_index)

    def _save_replay_name(self) -> None:
        try:
            replay_name = ""
            combo_model = self._labels.get("quest_replay_combo_model")
            if combo_model is not None:
                selected_index = int(combo_model.get_item_value_model().as_int)
                if 0 <= selected_index < len(self._quest_recording_names):
                    replay_name = self._quest_recording_names[selected_index]
            if not replay_name:
                raise ValueError("select a replay name")
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

    def _save_record_name(self) -> None:
        try:
            record_name = str(self._labels["quest_record_model"].get_value_as_string()).strip()
            if not record_name:
                raise ValueError("record name must not be empty")
            control = self._read_flow_control()
            control["record_name"] = record_name
            self._write_flow_control(control)
            self._labels["quest_status"].text = f"Saved record={record_name}. Live recording is armed now; press A to start."
        except Exception as exc:
            self._labels["quest_status"].text = f"Save failed: {exc}"

    def _delete_selected_replay(self) -> None:
        try:
            combo_model = self._labels.get("quest_replay_combo_model")
            if combo_model is None:
                raise ValueError("replay dropdown unavailable")
            selected_index = int(combo_model.get_item_value_model().as_int)
            if not (0 <= selected_index < len(self._quest_recording_names)):
                raise ValueError("select a replay to delete")
            replay_name = self._quest_recording_names[selected_index]
            replay_path = RECORDINGS_DIR / f"{replay_name}.ndjson"
            if not replay_path.exists():
                raise ValueError(f"recording not found: {replay_name}")
            replay_path.unlink()
            control = self._read_flow_control()
            if str(control.get("replay_name") or "") == replay_name:
                control["replay_name"] = ""
                self._write_flow_control(control)
                self._last_saved_replay_name = ""
            remaining = self._list_recording_names()
            next_selected = remaining[0] if remaining else None
            self._refresh_replay_dropdown(next_selected)
            self._labels["quest_status"].text = f"Deleted replay={replay_name}."
        except Exception as exc:
            self._labels["quest_status"].text = f"Delete failed: {exc}"

    def _list_joint_names(self) -> list[str]:
        config = self._read_vivy_config()
        names = config.get("joint_names")
        if isinstance(names, list):
            return [str(name) for name in names]
        return []

    def _joint_axis_map(self) -> dict[str, str]:
        config = self._read_vivy_config()
        joints = dict(config.get("joints") or {})
        result: dict[str, str] = {}
        for name in self._list_joint_names():
            result[name] = str(dict(joints.get(name) or {}).get("axis") or "-")
        return result

    def _refresh_ik_jacobian_mode_dropdown(self, selected_mode: str | None = None) -> None:
        try:
            import omni.ui as ui
        except ImportError:
            return
        container = self._labels.get("ik_jacobian_mode_container")
        if container is None:
            return
        try:
            container.clear()
        except Exception:
            pass
        with container:
            combo = ui.ComboBox(0, *self._ik_jacobian_modes, width=160)
            self._labels["ik_jacobian_mode_combo"] = combo
            self._labels["ik_jacobian_mode_combo_model"] = combo.model
        if selected_mode in self._ik_jacobian_modes:
            selected_index = self._ik_jacobian_modes.index(selected_mode)
        else:
            selected_index = 0
        self._labels["ik_jacobian_mode_combo_model"].get_item_value_model().set_value(selected_index)

    def _selected_ik_jacobian_mode(self) -> str:
        combo_model = self._labels.get("ik_jacobian_mode_combo_model")
        if combo_model is None:
            return "finite_difference"
        selected_index = int(combo_model.get_item_value_model().as_int)
        if 0 <= selected_index < len(self._ik_jacobian_modes):
            return self._ik_jacobian_modes[selected_index]
        return "finite_difference"

    def _save_joint_mode_axis(self, joint_name: str, *, mode: str | None = None, axis: str | None = None, weight: float | None = None) -> None:
        try:
            config = self._read_vivy_config()
            joints = dict(config.get("joints") or {})
            joint_entry = dict(joints.get(joint_name) or {})
            if mode is not None:
                if mode not in {"hold", "solve"}:
                    raise ValueError("mode must be hold or solve")
                joint_entry["hold_start"] = mode == "hold"
            if axis is not None:
                if axis not in self._joint_axis_options:
                    raise ValueError("invalid axis")
                joint_entry["axis"] = axis
            if weight is not None:
                if weight < 0.0:
                    raise ValueError("weight must be >= 0")
                joint_entry["weight"] = float(weight)
            joints[joint_name] = joint_entry
            config["joints"] = joints
            self._write_vivy_config(config)
            mode_text = "hold" if bool(joint_entry.get("hold_start", False)) else "solve"
            axis_text = str(joint_entry.get("axis") or "-")
            weight_text = float(joint_entry.get("weight", 1.0))
            self._labels["ik_status"].text = f"Saved {joint_name}: axis={axis_text} mode={mode_text} weight={weight_text:.2f}. Applies live."
        except Exception as exc:
            self._labels["ik_status"].text = f"Save failed: {exc}"

    def _refresh_ik_joint_table(self, rows: dict[str, dict[str, Any]]) -> None:
        try:
            import omni.ui as ui
        except ImportError:
            return
        container = self._labels.get("ik_joint_table_container")
        if container is None:
            return
        names = self._list_joint_names()
        self._ik_joint_names = names
        axis_map = self._joint_axis_map()
        config = self._read_vivy_config()
        joints = dict(config.get("joints") or {})
        signature = tuple((joint_name, str(axis_map.get(joint_name, "Y")), str(rows.get(joint_name, {}).get("mode") or "solve"), float(dict(joints.get(joint_name) or {}).get("weight", 1.0))) for joint_name in names)
        if signature == self._last_ik_table_signature:
            return
        self._last_ik_table_signature = signature
        try:
            container.clear()
        except Exception:
            pass
        header_style = {"color": self._TEXT_NEUTRAL, "font_size": 12}
        with container:
            with ui.HStack(height=22, spacing=8):
                ui.Label("joint", width=170, style=header_style)
                ui.Label("axis", width=90, style=header_style)
                ui.Label("weight", width=70, style=header_style)
                ui.Label("mode", width=90, style=header_style)
            for joint_name in names:
                current_axis = str(axis_map.get(joint_name, "Y"))
                current_mode = str(rows.get(joint_name, {}).get("mode") or "solve")
                current_weight = float(dict(joints.get(joint_name) or {}).get("weight", 1.0))
                with ui.HStack(height=24, spacing=8):
                    ui.Label(joint_name, width=170, style={"color": self._TEXT_NEUTRAL, "font_size": 12})
                    axis_index = self._joint_axis_options.index(current_axis) if current_axis in self._joint_axis_options else 0
                    axis_combo = ui.ComboBox(axis_index, *self._joint_axis_options, width=90)
                    axis_model = axis_combo.model
                    axis_item_model = axis_model.get_item_value_model()
                    axis_item_model.add_value_changed_fn(
                        lambda model, joint_name=joint_name: self._save_joint_mode_axis(
                            joint_name,
                            axis=self._joint_axis_options[int(model.as_int)],
                        )
                    )
                    weight_field = ui.StringField(width=70)
                    weight_field.model.set_value(f"{current_weight:.2f}")
                    weight_field.model.add_end_edit_fn(
                        lambda model, joint_name=joint_name: self._save_joint_mode_axis(
                            joint_name,
                            weight=float(model.get_value_as_string()),
                        )
                    )
                    mode_options = ["solve", "hold"]
                    mode_index = mode_options.index(current_mode) if current_mode in mode_options else 0
                    mode_combo = ui.ComboBox(mode_index, *mode_options, width=90)
                    mode_model = mode_combo.model
                    mode_item_model = mode_model.get_item_value_model()
                    mode_item_model.add_value_changed_fn(
                        lambda model, joint_name=joint_name: self._save_joint_mode_axis(
                            joint_name,
                            mode=["solve", "hold"][int(model.as_int)],
                        )
                    )

    def _save_ik_tuning(self) -> None:
        try:
            config = self._read_vivy_config()
            controller_defaults = dict(config.get("controller_defaults") or {})
            controller_defaults["ik_rate_hz"] = float(self._labels["ik_rate_hz_model"].get_value_as_string())
            jacobian_mode = self._selected_ik_jacobian_mode()
            controller_defaults["ik_jacobian_mode"] = jacobian_mode
            controller_defaults["ik_max_iteration"] = int(self._labels["ik_max_iteration_model"].get_value_as_string())
            controller_defaults["ik_damping"] = float(self._labels["ik_damping_model"].get_value_as_string())
            controller_defaults["ik_max_step_deg"] = float(self._labels["ik_max_step_model"].get_value_as_string())
            controller_defaults["output_max_delta_deg_per_tick"] = float(
                self._labels["output_max_delta_model"].get_value_as_string()
            )
            config["controller_defaults"] = controller_defaults
            self._write_vivy_config(config)
            self._labels["ik_status"].text = (
                "Saved IK tuning. Applies live and persists for next run."
            )
        except Exception as exc:
            self._labels["ik_status"].text = f"Save failed: {exc}"

    @staticmethod
    def _current_payload_rows(payload: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
        payload = payload or {}
        rows = payload.get("joint_display_rows") or []
        return {str(row.get("joint")): row for row in rows if isinstance(row, dict)}

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
                    self._labels["jump_to_source_button"] = ui.Button(
                        "Go to source output",
                        height=28,
                        clicked_fn=lambda: self._jump_to_source_output(),
                    )
                    self._labels["sim_toggle_button"] = ui.Button(
                        "Toggle Sim View",
                        height=28,
                        clicked_fn=lambda: self._toggle_sim_view(),
                    )
                    self._labels["pitch_frames_button"] = ui.Button(
                        "Toggle Pitch Frames",
                        height=28,
                        clicked_fn=lambda: self._toggle_pitch_frames(),
                    )
                    self._quest_section.build(ui, self._window.frame, header_style, value_style)
                    self._mapping_section.build(ui, self._window.frame, header_style, value_style)
                    self._ik_section.build(ui, self._window.frame, header_style, value_style)
                    try:
                        self._labels["quest_editor"].visible = False
                        self._labels["axis_editor"].visible = False
                        self._labels["ik_editor"].visible = False
                        self._labels["jump_to_source_button"].visible = False
                        self._labels["pitch_frames_button"].visible = False
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
        self._last_payload_rows = self._current_payload_rows(payload)
        selected = str(flow_state.get("selected_node") or "sim")
        self._labels["selected"].text = selected
        if selected != self._last_selected:
            if selected == "ik":
                self._last_ik_table_signature = None
                self._ik_section.load_from_config()
            elif selected == "processor":
                self._mapping_section.load_from_config()
            self._last_selected = selected

        waiting_for_anchor = bool(payload.get("waiting_for_anchor", True))
        freeze_active = bool(payload.get("freeze_active", False))
        recording_name = str(payload.get("recording_name") or "-")
        recording_status = str(payload.get("recording_status") or "idle")
        recording_packet_count = int(payload.get("recording_packet_count") or 0)
        sim_view_enabled = bool(flow_state.get("sim_view_enabled", True))
        show_pitch_frames = bool(flow_state.get("show_pitch_frames", False))
        replay_name = str(flow_state.get("replay_name") or "-")
        record_name = str(flow_state.get("record_name") or "-")
        quest_mode = str(flow_state.get("quest_mode") or "?")
        robot_output = str(flow_state.get("robot_output") or "-")

        if selected == "sim":
            self._labels["detail"].text = (
                f"Sim target marker branch\n"
                f"enabled={sim_view_enabled} pitch_frames={show_pitch_frames}"
            )
            self._labels["action_hint"].text = "Use the buttons below to toggle sim view and pitch-joint frames."
            try:
                self._labels["sim_toggle_button"].visible = True
                self._labels["sim_toggle_button"].text = "Turn Sim View Off" if sim_view_enabled else "Turn Sim View On"
                self._labels["pitch_frames_button"].visible = True
                self._labels["pitch_frames_button"].text = "Hide Pitch Frames" if show_pitch_frames else "Show Pitch Frames"
                self._labels["jump_to_source_button"].visible = False
                self._labels["quest_editor"].visible = False
                self._labels["axis_editor"].visible = False
                self._labels["ik_editor"].visible = False
            except Exception:
                pass
        else:
            detail = {
                "quest": f"quest_mode={quest_mode} replay={replay_name} record={record_name} recording={recording_status}",
                "processor": "Processes Quest packets into teleop state",
                "teleop_state": f"waiting_for_anchor={waiting_for_anchor} freeze_active={freeze_active}",
                "ik": "Quest IK teleoperator branch",
                "quest_anchor_capture": "Place the Quest controller where you want, press A, and that current controller position/orientation plus the current robot pose become the new anchor.",
                "anchor_delta": "Computes current hand position minus anchor position.",
                "deadband": "Suppresses small motion below the configured threshold.",
                "processor": "Processes Quest packets and applies axis/sign remap plus target conditioning before IK.",
                "axis_remap": "Applies axis reorder and sign flips to Quest deltas.",
                "world_transform": "Applies scale and world-frame rotation.",
                "target_pose": "Builds the target pose used for downstream consumers.",
                "joint_targets": "Final IK output: computed joint-angle targets for downstream consumers.",
                "fanout": "Consumes joint-angle targets for real/log sinks",
                "real": "Hardware sink branch",
                "log": f"log sink branch output={robot_output}",
                "sim_input": "Red-dot / target visualization from teleop state.",
                "sim_joint_targets": "Arm motion from IK joint targets.",
            }.get(selected, "Select a node from the flow tree.")
            self._labels["detail"].text = detail
            source_label = self._INPUT_SOURCE_LABEL.get(selected)
            self._labels["action_hint"].text = f"Jump to {source_label}." if source_label else ""
            try:
                self._labels["sim_toggle_button"].visible = False
                self._labels["pitch_frames_button"].visible = False
                self._labels["jump_to_source_button"].visible = source_label is not None
                if source_label is not None:
                    self._labels["jump_to_source_button"].text = f"Go to {source_label}"
                self._labels["quest_editor"].visible = selected == "quest"
                self._labels["axis_editor"].visible = selected == "processor"
                self._labels["ik_editor"].visible = selected == "ik"
            except Exception:
                pass
            if selected == "quest":
                try:
                    self._quest_section.update(
                        replay_name=replay_name,
                        record_name=record_name,
                        recording_name=recording_name,
                        recording_status=recording_status,
                        recording_packet_count=recording_packet_count,
                    )
                except Exception:
                    pass
            elif selected == "ik":
                try:
                    self._ik_section.update(dict(self._last_payload_rows), dict(payload))
                except Exception:
                    pass
