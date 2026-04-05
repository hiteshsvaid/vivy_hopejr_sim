from __future__ import annotations

from typing import Any


class IkDetailSection:
    def __init__(self, panel: Any):
        self.panel = panel

    def build(self, ui: Any, parent: Any, header_style: dict[str, Any], value_style: dict[str, Any]) -> None:
        helper_style = {**value_style, "font_size": 12}
        with ui.VStack(spacing=4) as ik_editor:
            self.panel._labels["ik_editor"] = ik_editor
            ui.Label("IK Hold Control", style=header_style)
            with ui.HStack(height=26, spacing=6) as ik_joint_container:
                self.panel._ik_joint_container = ik_joint_container
                self.panel._refresh_hold_dropdown()
            self.panel._labels["ik_toggle_button"] = ui.Button(
                "Set Hold",
                height=28,
                clicked_fn=lambda: self.panel._toggle_hold_selected_joint(),
            )
            ui.Spacer(height=6)
            ui.Label("IK Tuning", style=header_style)

            with ui.VStack(spacing=2):
                with ui.HStack(height=26, spacing=6):
                    ui.Label("ik_damping", width=100, style=value_style)
                    ik_damping = ui.StringField(width=100)
                    self.panel._labels["ik_damping_model"] = ik_damping.model
                ui.Label(
                    "DLS damping inside the IK solve. Higher values make the solve more conservative.",
                    style=helper_style,
                    word_wrap=True,
                )

            with ui.VStack(spacing=2):
                with ui.HStack(height=26, spacing=6):
                    ui.Label("ik_max_step_deg", width=100, style=value_style)
                    ik_step = ui.StringField(width=100)
                    self.panel._labels["ik_max_step_model"] = ik_step.model
                ui.Label(
                    "Maximum joint change the IK solver may propose in one solve step.",
                    style=helper_style,
                    word_wrap=True,
                )

            with ui.VStack(spacing=2):
                with ui.HStack(height=26, spacing=6):
                    ui.Label("output_max_delta_deg_per_tick", width=170, style=value_style)
                    output_delta = ui.StringField(width=100)
                    self.panel._labels["output_max_delta_model"] = output_delta.model
                ui.Label(
                    "Post-IK output clamp. Limits how much the commanded joint target may change per control tick.",
                    style=helper_style,
                    word_wrap=True,
                )

            self.panel._labels["ik_tuning_button"] = ui.Button(
                "Save IK Tuning",
                height=28,
                clicked_fn=lambda: self.panel._save_ik_tuning(),
            )
            self.panel._labels["ik_status"] = ui.Label("", style=value_style, word_wrap=True)

    def load_from_config(self) -> None:
        config = self.panel._read_vivy_config()
        controller_defaults = dict(config.get("controller_defaults") or {})
        self.panel._labels["ik_damping_model"].set_value(str(controller_defaults.get("ik_damping", 0.01)))
        self.panel._labels["ik_max_step_model"].set_value(str(controller_defaults.get("ik_max_step_deg", 8.0)))
        self.panel._labels["output_max_delta_model"].set_value(
            str(controller_defaults.get("output_max_delta_deg_per_tick", 2.0))
        )

    def update(self, rows: dict[str, dict[str, Any]]) -> None:
        names = self.panel._list_joint_names()
        current_joint = self.panel._selected_hold_joint()
        if names != self.panel._ik_joint_names:
            self.panel._refresh_hold_dropdown(current_joint)
            current_joint = self.panel._selected_hold_joint()
        row = rows.get(str(current_joint or ""), {})
        current_mode = str(row.get("mode") or "solve")
        self.panel._labels["ik_toggle_button"].text = "Set Solve" if current_mode == "hold" else "Set Hold"
        self.panel._labels["ik_status"].text = (
            f"Selected joint={current_joint or '-'} current_mode={current_mode}. "
            "Saves to main config and applies live."
        )
