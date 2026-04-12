from __future__ import annotations

from typing import Any


class IkDetailSection:
    def __init__(self, panel: Any):
        self.panel = panel

    def build(self, ui: Any, parent: Any, header_style: dict[str, Any], value_style: dict[str, Any]) -> None:
        helper_style = {**value_style, "font_size": 12}
        with ui.VStack(spacing=4) as ik_editor:
            self.panel._labels["ik_editor"] = ik_editor
            ui.Label("IK Joint Table", style=header_style)
            with ui.VStack(spacing=4) as ik_joint_table_container:
                self.panel._labels["ik_joint_table_container"] = ik_joint_table_container
                with ui.HStack(height=22, spacing=8):
                    ui.Label("joint", width=170, style={**value_style, "font_size": 12})
                    ui.Label("axis", width=110, style={**value_style, "font_size": 12})
                    ui.Label("mode", width=90, style={**value_style, "font_size": 12})
            ui.Spacer(height=6)
            ui.Label("IK Tuning", style=header_style)

            with ui.VStack(spacing=2):
                with ui.HStack(height=26, spacing=6):
                    ui.Label("ik_rate_hz", width=120, style=value_style)
                    ik_rate_hz = ui.StringField(width=100)
                    self.panel._labels["ik_rate_hz_model"] = ik_rate_hz.model
                ui.Label(
                    "Fixed IK control-loop rate in Hz. Quest input updates asynchronously and IK solves at this rate.",
                    style=helper_style,
                    word_wrap=True,
                )

            with ui.VStack(spacing=2):
                with ui.HStack(height=26, spacing=6):
                    ui.Label("ik_actual_hz", width=120, style=value_style)
                    self.panel._labels["ik_actual_hz_value"] = ui.Label("-", style=value_style)
                ui.Label(
                    "Measured IK loop rate from the current run. This is runtime status only.",
                    style=helper_style,
                    word_wrap=True,
                )

            with ui.VStack(spacing=2):
                with ui.HStack(height=26, spacing=6):
                    ui.Label("ik_jacobian_mode", width=120, style=value_style)
                    jacobian_mode_container = ui.HStack(width=160, spacing=0)
                    self.panel._labels["ik_jacobian_mode_container"] = jacobian_mode_container
                    self.panel._refresh_ik_jacobian_mode_dropdown()
                ui.Label(
                    "Jacobian implementation used by IK. Use finite_difference for reference and analytic for speed.",
                    style=helper_style,
                    word_wrap=True,
                )

            with ui.VStack(spacing=2):
                with ui.HStack(height=26, spacing=6):
                    ui.Label("ik_max_iteration", width=120, style=value_style)
                    ik_max_iteration = ui.StringField(width=100)
                    self.panel._labels["ik_max_iteration_model"] = ik_max_iteration.model
                ui.Label(
                    "Maximum number of IK iterations per control tick. Lower values trade accuracy for loop rate.",
                    style=helper_style,
                    word_wrap=True,
                )

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

            with ui.VStack(spacing=2):
                with ui.HStack(height=26, spacing=6):
                    ui.Label("forearm_twist_rotation", width=170, style=value_style)
                    forearm_enable = ui.StringField(width=100)
                    self.panel._labels["forearm_twist_enable_model"] = forearm_enable.model
                ui.Label(
                    "Enable controller-rotation-driven forearm twist. Use true or false.",
                    style=helper_style,
                    word_wrap=True,
                )

            with ui.VStack(spacing=2):
                with ui.HStack(height=26, spacing=6):
                    ui.Label("forearm_twist_axis", width=170, style=value_style)
                    forearm_axis = ui.StringField(width=100)
                    self.panel._labels["forearm_twist_axis_model"] = forearm_axis.model
                ui.Label(
                    "Controller rotation axis used for forearm twist. Use x, y, or z.",
                    style=helper_style,
                    word_wrap=True,
                )

            with ui.VStack(spacing=2):
                with ui.HStack(height=26, spacing=6):
                    ui.Label("forearm_twist_sign", width=170, style=value_style)
                    forearm_sign = ui.StringField(width=100)
                    self.panel._labels["forearm_twist_sign_model"] = forearm_sign.model
                ui.Label(
                    "Sign applied to the controller rotation axis for forearm twist. Usually 1 or -1.",
                    style=helper_style,
                    word_wrap=True,
                )

            with ui.VStack(spacing=2):
                with ui.HStack(height=26, spacing=6):
                    ui.Label("forearm_twist_scale", width=170, style=value_style)
                    forearm_scale = ui.StringField(width=100)
                    self.panel._labels["forearm_twist_scale_model"] = forearm_scale.model
                ui.Label(
                    "Scale multiplier from controller rotation to forearm twist degrees.",
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
        self.panel._labels["ik_rate_hz_model"].set_value(str(controller_defaults.get("ik_rate_hz", 30.0)))
        self.panel._refresh_ik_jacobian_mode_dropdown(
            str(controller_defaults.get("ik_jacobian_mode", "finite_difference"))
        )
        self.panel._labels["ik_max_iteration_model"].set_value(str(controller_defaults.get("ik_max_iteration", 80)))
        self.panel._labels["ik_damping_model"].set_value(str(controller_defaults.get("ik_damping", 0.01)))
        self.panel._labels["ik_max_step_model"].set_value(str(controller_defaults.get("ik_max_step_deg", 8.0)))
        self.panel._labels["output_max_delta_model"].set_value(
            str(controller_defaults.get("output_max_delta_deg_per_tick", 2.0))
        )
        self.panel._labels["forearm_twist_enable_model"].set_value(
            str(bool(controller_defaults.get("forearm_twist_from_controller_rotation", False))).lower()
        )
        self.panel._labels["forearm_twist_axis_model"].set_value(
            str(controller_defaults.get("forearm_twist_controller_axis", "z"))
        )
        self.panel._labels["forearm_twist_sign_model"].set_value(
            str(controller_defaults.get("forearm_twist_controller_sign", 1.0))
        )
        self.panel._labels["forearm_twist_scale_model"].set_value(
            str(controller_defaults.get("forearm_twist_controller_scale", 1.0))
        )

    def update(self, rows: dict[str, dict[str, Any]], payload: dict[str, Any]) -> None:
        names = self.panel._list_joint_names()
        self.panel._refresh_ik_joint_table(rows)
        ik_actual_hz = float(payload.get("ik_actual_hz") or 0.0)
        self.panel._labels["ik_actual_hz_value"].text = f"{ik_actual_hz:.1f}"
        jacobian_mode = str(payload.get("ik_jacobian_mode") or self.panel._selected_ik_jacobian_mode() or "-")
        self.panel._labels["ik_status"].text = (
            f"joints={len(names)} jacobian={jacobian_mode}. Axis and mode changes save to main config and apply live."
        )
