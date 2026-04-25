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
        position_scale = float(script_defaults.get("position_scale", controller_defaults.get("position_scale", 1.0)))
        try:
            self._labels["axis_axes_model"].set_value(axes)
            self._labels["axis_sign_x_model"].set_value(str(signs[0]))
            self._labels["axis_sign_y_model"].set_value(str(signs[1]))
            self._labels["axis_sign_z_model"].set_value(str(signs[2]))
            self._labels["position_scale_model"].set_value(str(position_scale))
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
            position_scale = float(self._labels["position_scale_model"].get_value_as_string())
            target_max_delta = float(self._labels["target_max_delta_model"].get_value_as_string())
            controller_defaults["position_scale"] = position_scale
            controller_defaults["quest_position_axes"] = axes
            controller_defaults["quest_position_signs"] = list(signs)
            controller_defaults["target_max_delta_m_per_tick"] = target_max_delta
            script_defaults["position_scale"] = position_scale
            script_defaults["quest_position_axes"] = axes
            script_defaults["quest_position_signs"] = list(signs)
            config["controller_defaults"] = controller_defaults
            config["script_editor_test_defaults"] = script_defaults
            self._write_vivy_config(config)
            self._labels["axis_status"].text = (
                f"Saved position_scale={position_scale} axes={axes} signs={signs} "
                f"target_max_delta={target_max_delta}. Applies live."
            )
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

    def _list_ik_table_joint_names(self) -> list[str]:
        config = self._read_vivy_config()
        names = list(self._list_joint_names())
        excluded = ((config.get("kinematics") or {}).get("excluded_joints") or [])
        for name in excluded:
            name_text = str(name)
            if name_text not in names:
                names.append(name_text)
        return names

    def _is_ik_joint(self, joint_name: str) -> bool:
        return joint_name in set(self._list_joint_names())

    def _joint_axis_map(self) -> dict[str, str]:
        config = self._read_vivy_config()
        joints = dict(config.get("joints") or {})
        result: dict[str, str] = {}
        for name in self._list_ik_table_joint_names():
            result[name] = str(dict(joints.get(name) or {}).get("axis") or "Y")
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

    @staticmethod
    def _parse_bool_text(value: str) -> bool:
        cleaned = str(value).strip().lower()
        if cleaned in {"1", "true", "yes", "on"}:
            return True
        if cleaned in {"0", "false", "no", "off"}:
            return False
        raise ValueError("expected true/false")

    @staticmethod
    def _parse_scale_text(value: str) -> float:
        cleaned = str(value).strip().lower().replace("deg", "").replace("x", "").strip()
        return float(cleaned)

    def _save_joint_mode_axis(
        self,
        joint_name: str,
        *,
        mode: str | None = None,
        axis: str | None = None,
        weight: float | None = None,
        neutral_bias_weight: float | None = None,
        output_max_delta_deg_per_tick: float | None = None,
    ) -> None:
        try:
            config = self._read_vivy_config()
            joints = dict(config.get("joints") or {})
            joint_entry = dict(joints.get(joint_name) or {})
            if mode is not None:
                if not self._is_ik_joint(joint_name):
                    self._labels["ik_status"].text = f"Ignored {joint_name}: non-IK joints do not use solve/hold."
                    return
                if mode not in {"hold", "solve", "direct"}:
                    raise ValueError("mode must be solve, hold, or direct")
                joint_entry["ik_mode"] = mode
                joint_entry["hold_start"] = mode == "hold"
            if axis is not None:
                if axis not in self._joint_axis_options:
                    raise ValueError("invalid axis")
                joint_entry["axis"] = axis
            if weight is not None:
                if weight < 0.0:
                    raise ValueError("weight must be >= 0")
                joint_entry["weight"] = float(weight)
            if neutral_bias_weight is not None:
                if neutral_bias_weight < 0.0:
                    raise ValueError("neutral_bias_weight must be >= 0")
                joint_entry["neutral_bias_weight"] = float(neutral_bias_weight)
            if output_max_delta_deg_per_tick is not None:
                if output_max_delta_deg_per_tick < 0.0:
                    raise ValueError("output_max_delta_deg_per_tick must be >= 0")
                joint_entry["output_max_delta_deg_per_tick"] = float(output_max_delta_deg_per_tick)
            joints[joint_name] = joint_entry
            config["joints"] = joints
            self._write_vivy_config(config)
            mode_text = str(joint_entry.get("ik_mode") or ("hold" if bool(joint_entry.get("hold_start", False)) else "solve"))
            axis_text = str(joint_entry.get("axis") or "-")
            weight_text = float(joint_entry.get("weight", 1.0))
            neutral_bias_text = float(joint_entry.get("neutral_bias_weight", 0.0))
            tick_text = float(
                joint_entry.get(
                    "output_max_delta_deg_per_tick",
                    (config.get("controller_defaults") or {}).get("output_max_delta_deg_per_tick", 0.0),
                )
            )
            self._labels["ik_status"].text = (
                f"Saved {joint_name}: axis={axis_text} mode={mode_text} "
                f"weight={weight_text:.2f} neutral_bias={neutral_bias_text:.2f} "
                f"joint_tick={tick_text:.2f}. Applies live."
            )
        except Exception as exc:
            self._labels["ik_status"].text = f"Save failed: {exc}"

    def _joint_direct_input_config(self, joint_name: str, controller_defaults: dict[str, Any]) -> dict[str, Any]:
        if joint_name == "right_forearm_twist":
            enabled = bool(controller_defaults.get("forearm_twist_from_controller_rotation", False))
            return {
                "source": "rotation" if enabled else "none",
                "axis": str(controller_defaults.get("forearm_twist_controller_axis", "z")).lower(),
                "sign": float(controller_defaults.get("forearm_twist_controller_sign", 1.0)),
                "scale": float(controller_defaults.get("forearm_twist_controller_scale", 1.0)),
                "deadband": None,
            }
        if joint_name == "right_wrist":
            return {
                "source": "thumbstick",
                "axis": "X",
                "sign": float(controller_defaults.get("right_wrist_thumbstick_sign", -1.0)),
                "scale": None,
                "deadband": float(controller_defaults.get("right_wrist_thumbstick_deadband", 0.1)),
            }
        if joint_name == "right_palm":
            return {
                "source": "thumbstick",
                "axis": "Y",
                "sign": float(controller_defaults.get("right_palm_thumbstick_sign", 1.0)),
                "scale": None,
                "deadband": float(controller_defaults.get("right_palm_thumbstick_deadband", 0.1)),
            }
        return {"source": "none", "axis": "-", "sign": None, "scale": None, "deadband": None}

    def _save_joint_direct_input(
        self,
        joint_name: str,
        *,
        source: str | None = None,
        input_axis: str | None = None,
        sign: float | None = None,
        scale: float | None = None,
        deadband: float | None = None,
    ) -> None:
        try:
            config = self._read_vivy_config()
            controller_defaults = dict(config.get("controller_defaults") or {})
            if joint_name == "right_forearm_twist":
                if source is not None:
                    if source not in {"none", "rotation"}:
                        raise ValueError("forearm input source must be none or rotation")
                    controller_defaults["forearm_twist_from_controller_rotation"] = source == "rotation"
                if input_axis is not None:
                    input_axis = input_axis.strip().lower()
                    if input_axis not in {"x", "y", "z"}:
                        raise ValueError("forearm input axis must be x, y, or z")
                    controller_defaults["forearm_twist_controller_axis"] = input_axis
                if sign is not None:
                    controller_defaults["forearm_twist_controller_sign"] = float(sign)
                if scale is not None:
                    controller_defaults["forearm_twist_controller_scale"] = float(scale)
            elif joint_name == "right_wrist":
                if sign is not None:
                    controller_defaults["right_wrist_thumbstick_sign"] = float(sign)
                if deadband is not None:
                    controller_defaults["right_wrist_thumbstick_deadband"] = float(deadband)
            elif joint_name == "right_palm":
                if sign is not None:
                    controller_defaults["right_palm_thumbstick_sign"] = float(sign)
                if deadband is not None:
                    controller_defaults["right_palm_thumbstick_deadband"] = float(deadband)
            else:
                self._labels["ik_status"].text = f"Ignored {joint_name}: no direct input mapping."
                return
            config["controller_defaults"] = controller_defaults
            self._write_vivy_config(config)
            direct = self._joint_direct_input_config(joint_name, controller_defaults)
            self._last_ik_table_signature = None
            self._labels["ik_status"].text = (
                f"Saved {joint_name} direct input: source={direct['source']} axis={direct['axis']} "
                f"sign={direct['sign']} scale={direct['scale']} deadband={direct['deadband']}."
            )
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
        names = self._list_ik_table_joint_names()
        self._ik_joint_names = names
        axis_map = self._joint_axis_map()
        config = self._read_vivy_config()
        joints = dict(config.get("joints") or {})
        controller_defaults = dict(config.get("controller_defaults") or {})
        default_output_delta = float(controller_defaults.get("output_max_delta_deg_per_tick", 2.0))
        ik_joint_names = set(self._list_joint_names())
        direct_inputs = {
            joint_name: self._joint_direct_input_config(joint_name, controller_defaults) for joint_name in names
        }
        signature = tuple(
            (
                joint_name,
                joint_name in ik_joint_names,
                str(axis_map.get(joint_name, "Y")),
                str(rows.get(joint_name, {}).get("mode") or "solve"),
                float(dict(joints.get(joint_name) or {}).get("weight", 1.0)),
                float(dict(joints.get(joint_name) or {}).get("neutral_bias_weight", 0.0)),
                float(dict(joints.get(joint_name) or {}).get("output_max_delta_deg_per_tick", default_output_delta)),
                str(direct_inputs[joint_name]["source"]),
                str(direct_inputs[joint_name]["axis"]),
                str(direct_inputs[joint_name]["sign"]),
                str(direct_inputs[joint_name]["scale"]),
                str(direct_inputs[joint_name]["deadband"]),
            )
            for joint_name in names
        )
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
                ui.Label("neutral bias", width=70, style=header_style)
                ui.Label("joint tick", width=70, style=header_style)
                ui.Label("IK mode", width=90, style=header_style)
                ui.Label("direct input", width=105, style=header_style)
                ui.Label("input axis", width=75, style=header_style)
                ui.Label("sign", width=60, style=header_style)
                ui.Label("scale *", width=70, style=header_style)
                ui.Label("deadband", width=70, style=header_style)
            for joint_name in names:
                joint_entry = dict(joints.get(joint_name) or {})
                is_ik_joint = joint_name in ik_joint_names
                direct_input = direct_inputs[joint_name]
                current_axis = str(axis_map.get(joint_name, "Y"))
                current_mode = str(
                    rows.get(joint_name, {}).get("mode")
                    or ("hold" if bool(joint_entry.get("hold_start", False)) else "solve")
                )
                current_weight = float(dict(joints.get(joint_name) or {}).get("weight", 1.0))
                current_neutral_bias = float(dict(joints.get(joint_name) or {}).get("neutral_bias_weight", 0.0))
                current_output_delta = float(joint_entry.get("output_max_delta_deg_per_tick", default_output_delta))
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
                    if is_ik_joint:
                        weight_field = ui.StringField(width=70)
                        weight_field.model.set_value(f"{current_weight:.2f}")
                        weight_field.model.add_end_edit_fn(
                            lambda model, joint_name=joint_name: self._save_joint_mode_axis(
                                joint_name,
                                weight=float(model.get_value_as_string()),
                            )
                        )
                        neutral_bias_field = ui.StringField(width=70)
                        neutral_bias_field.model.set_value(f"{current_neutral_bias:.2f}")
                        neutral_bias_field.model.add_end_edit_fn(
                            lambda model, joint_name=joint_name: self._save_joint_mode_axis(
                                joint_name,
                                neutral_bias_weight=float(model.get_value_as_string()),
                            )
                        )
                    else:
                        ui.Label("n/a", width=70, style={"color": self._TEXT_NEUTRAL, "font_size": 12})
                        ui.Label("n/a", width=70, style={"color": self._TEXT_NEUTRAL, "font_size": 12})
                    output_delta_field = ui.StringField(width=70)
                    output_delta_field.model.set_value(f"{current_output_delta:.2f}")
                    output_delta_field.model.add_end_edit_fn(
                        lambda model, joint_name=joint_name: self._save_joint_mode_axis(
                            joint_name,
                            output_max_delta_deg_per_tick=float(model.get_value_as_string()),
                        )
                    )
                    if is_ik_joint:
                        mode_options = ["solve", "hold", "direct"]
                        mode_index = mode_options.index(current_mode) if current_mode in mode_options else 0
                        mode_combo = ui.ComboBox(mode_index, *mode_options, width=90)
                        mode_model = mode_combo.model
                        mode_item_model = mode_model.get_item_value_model()
                        mode_item_model.add_value_changed_fn(
                            lambda model, joint_name=joint_name: self._save_joint_mode_axis(
                                joint_name,
                                mode=["solve", "hold", "direct"][int(model.as_int)],
                            )
                        )
                    else:
                        ui.Label(
                            "non-IK",
                            width=90,
                            style={"color": self._TEXT_NEUTRAL, "font_size": 12},
                        )
                    if joint_name == "right_forearm_twist":
                        source_options = ["none", "rotation"]
                        source_index = (
                            source_options.index(str(direct_input["source"]))
                            if str(direct_input["source"]) in source_options
                            else 0
                        )
                        source_combo = ui.ComboBox(source_index, *source_options, width=105)
                        source_model = source_combo.model.get_item_value_model()
                        source_model.add_value_changed_fn(
                            lambda model, joint_name=joint_name: self._save_joint_direct_input(
                                joint_name,
                                source=["none", "rotation"][int(model.as_int)],
                            )
                        )
                        input_axis_options = ["x", "y", "z"]
                        input_axis = str(direct_input["axis"]).lower()
                        input_axis_index = input_axis_options.index(input_axis) if input_axis in input_axis_options else 0
                        input_axis_combo = ui.ComboBox(input_axis_index, *input_axis_options, width=75)
                        input_axis_model = input_axis_combo.model.get_item_value_model()
                        input_axis_model.add_value_changed_fn(
                            lambda model, joint_name=joint_name: self._save_joint_direct_input(
                                joint_name,
                                input_axis=["x", "y", "z"][int(model.as_int)],
                            )
                        )
                        sign_field = ui.StringField(width=60)
                        sign_field.model.set_value(f"{float(direct_input['sign']):.2f}")
                        sign_field.model.add_end_edit_fn(
                            lambda model, joint_name=joint_name: self._save_joint_direct_input(
                                joint_name,
                                sign=float(model.get_value_as_string()),
                            )
                        )
                        scale_field = ui.StringField(width=70)
                        scale_field.model.set_value(f"{float(direct_input['scale']):.2f}x")
                        scale_field.model.add_end_edit_fn(
                            lambda model, joint_name=joint_name: self._save_joint_direct_input(
                                joint_name,
                                scale=self._parse_scale_text(model.get_value_as_string()),
                            )
                        )
                        ui.Label("n/a", width=70, style={"color": self._TEXT_NEUTRAL, "font_size": 12})
                    elif joint_name in {"right_wrist", "right_palm"}:
                        ui.Label("thumbstick", width=105, style={"color": self._TEXT_NEUTRAL, "font_size": 12})
                        ui.Label(str(direct_input["axis"]), width=75, style={"color": self._TEXT_NEUTRAL, "font_size": 12})
                        sign_field = ui.StringField(width=60)
                        sign_field.model.set_value(f"{float(direct_input['sign']):.2f}")
                        sign_field.model.add_end_edit_fn(
                            lambda model, joint_name=joint_name: self._save_joint_direct_input(
                                joint_name,
                                sign=float(model.get_value_as_string()),
                            )
                        )
                        ui.Label("joint tick", width=70, style={"color": self._TEXT_NEUTRAL, "font_size": 12})
                        deadband_field = ui.StringField(width=70)
                        deadband_field.model.set_value(f"{float(direct_input['deadband']):.2f}")
                        deadband_field.model.add_end_edit_fn(
                            lambda model, joint_name=joint_name: self._save_joint_direct_input(
                                joint_name,
                                deadband=float(model.get_value_as_string()),
                            )
                        )
                    else:
                        ui.Label("none", width=105, style={"color": self._TEXT_NEUTRAL, "font_size": 12})
                        ui.Label("n/a", width=75, style={"color": self._TEXT_NEUTRAL, "font_size": 12})
                        ui.Label("n/a", width=60, style={"color": self._TEXT_NEUTRAL, "font_size": 12})
                        ui.Label("n/a", width=70, style={"color": self._TEXT_NEUTRAL, "font_size": 12})
                        ui.Label("n/a", width=70, style={"color": self._TEXT_NEUTRAL, "font_size": 12})

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
            controller_defaults["ignore_ik_when_thumbstick_active"] = bool(
                self._labels["ignore_ik_when_thumbstick_active_model"].get_value_as_bool()
            )
            controller_defaults["thumbstick_release_deadband"] = float(
                self._labels["thumbstick_release_deadband_model"].get_value_as_string()
            )
            controller_defaults["thumbstick_release_hold_frames"] = int(
                self._labels["thumbstick_release_hold_frames_model"].get_value_as_string()
            )
            controller_defaults["thumbstick_release_target_move_tolerance_m"] = float(
                self._labels["thumbstick_release_target_move_tolerance_m_model"].get_value_as_string()
            )
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
        try:
            self._window.deferred_dock_in("Console")
        except Exception:
            pass
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
