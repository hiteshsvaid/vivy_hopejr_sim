from __future__ import annotations

from typing import Any


class IkDetailSection:
    def __init__(self, panel: Any):
        self.panel = panel

    def build(self, ui: Any, parent: Any, header_style: dict[str, Any], value_style: dict[str, Any]) -> None:
        helper_style = {**value_style, "font_size": 12}
        with ui.VStack(spacing=4) as ik_editor:
            self.panel._labels["ik_editor"] = ik_editor
            ui.Label("Joint Tuning Table", style=header_style)
            with ui.VStack(spacing=2):
                with ui.HStack(height=26, spacing=6):
                    ui.Label("global joint tick", width=170, style=value_style)
                    output_delta = ui.StringField(width=100)
                    self.panel._labels["output_max_delta_model"] = output_delta.model
                ui.Label(
                    "Default post-IK output clamp in deg/tick. Per-joint values below override this default.",
                    style=helper_style,
                    word_wrap=True,
                )
            with ui.VStack(spacing=4) as ik_joint_table_container:
                self.panel._labels["ik_joint_table_container"] = ik_joint_table_container
                with ui.HStack(height=22, spacing=8):
                    ui.Label("joint", width=170, style={**value_style, "font_size": 12})
                    ui.Label("axis", width=90, style={**value_style, "font_size": 12})
                    ui.Label("weight", width=70, style={**value_style, "font_size": 12})
                    ui.Label("neutral bias", width=70, style={**value_style, "font_size": 12})
                    ui.Label("joint tick", width=70, style={**value_style, "font_size": 12})
                    ui.Label("IK mode", width=90, style={**value_style, "font_size": 12})
                    ui.Label("direct input", width=105, style={**value_style, "font_size": 12})
                    ui.Label("input axis", width=75, style={**value_style, "font_size": 12})
                    ui.Label("sign", width=60, style={**value_style, "font_size": 12})
                    ui.Label("scale *", width=70, style={**value_style, "font_size": 12})
                    ui.Label("deadband", width=70, style={**value_style, "font_size": 12})
                ui.Label(
                    "* scale multiplies raw input. Rotation input is already degrees, so forearm scale is x. "
                    "Thumbstick input is -1..1, so wrist/palm scale is deg at full stick.",
                    style=helper_style,
                    word_wrap=True,
                )
            ui.Spacer(height=6)
            ui.Label("IK Tuning", style=header_style)

            with ui.VStack(spacing=2):
                with ui.HStack(height=26, spacing=6):
                    ui.Label("ik_rate_hz", width=120, style=value_style)
                    ik_rate_hz = ui.StringField(width=100)
                    self.panel._labels["ik_rate_hz_model"] = ik_rate_hz.model
                    ui.Label("actual", width=52, style=value_style)
                    self.panel._labels["ik_actual_hz_value"] = ui.Label("-", width=70, style=value_style)
                ui.Label(
                    "Target IK control-loop rate and measured runtime rate in Hz.",
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
                    ui.Label("thumbstick skips IK", width=120, style=value_style)
                    thumbstick_ignore = ui.CheckBox(width=24)
                    self.panel._labels["ignore_ik_when_thumbstick_active_model"] = thumbstick_ignore.model
                ui.Label(
                    "When enabled, active wrist or palm thumbstick input bypasses pose IK and only applies direct thumbstick joint commands.",
                    style=helper_style,
                    word_wrap=True,
                )

            ui.Spacer(height=6)
            ui.Label("Thumbstick Handoff", style=header_style)

            with ui.VStack(spacing=2):
                with ui.HStack(height=26, spacing=6):
                    ui.Label("release deadband", width=120, style=value_style)
                    release_deadband = ui.StringField(width=100)
                    self.panel._labels["thumbstick_release_deadband_model"] = release_deadband.model
                ui.Label(
                    "Stick-center threshold used to keep thumbstick mode active during release. IK stays suppressed until the active axis is near center.",
                    style=helper_style,
                    word_wrap=True,
                )

            with ui.VStack(spacing=2):
                with ui.HStack(height=26, spacing=6):
                    ui.Label("release hold frames", width=120, style=value_style)
                    release_hold_frames = ui.StringField(width=100)
                    self.panel._labels["thumbstick_release_hold_frames_model"] = release_hold_frames.model
                ui.Label(
                    "Short solve-joint hold after thumbstick release. This prevents the first IK frame from jumping immediately.",
                    style=helper_style,
                    word_wrap=True,
                )

            with ui.VStack(spacing=2):
                with ui.HStack(height=26, spacing=6):
                    ui.Label("target move tol (m)", width=120, style=value_style)
                    release_move_tol = ui.StringField(width=100)
                    self.panel._labels["thumbstick_release_target_move_tolerance_m_model"] = release_move_tol.model
                ui.Label(
                    "After thumbstick release, keep solve joints locked until the cartesian target moves by at least this amount. Larger values resist small controller drift.",
                    style=helper_style,
                    word_wrap=True,
                )

            ui.Label(
                "Current solution: wrist and palm stay direct-input only, IK stays off while thumbstick is active, and proximal solve joints remain locked through release until the target meaningfully moves.",
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
        self.panel._labels["ignore_ik_when_thumbstick_active_model"].set_value(
            bool(controller_defaults.get("ignore_ik_when_thumbstick_active", True))
        )
        self.panel._labels["thumbstick_release_deadband_model"].set_value(
            str(controller_defaults.get("thumbstick_release_deadband", 0.1))
        )
        self.panel._labels["thumbstick_release_hold_frames_model"].set_value(
            str(controller_defaults.get("thumbstick_release_hold_frames", 2))
        )
        self.panel._labels["thumbstick_release_target_move_tolerance_m_model"].set_value(
            str(controller_defaults.get("thumbstick_release_target_move_tolerance_m", 0.015))
        )
        self.panel._labels["output_max_delta_model"].set_value(
            str(controller_defaults.get("output_max_delta_deg_per_tick", 2.0))
        )

    def update(self, rows: dict[str, dict[str, Any]], payload: dict[str, Any]) -> None:
        names = self.panel._list_ik_table_joint_names()
        self.panel._refresh_ik_joint_table(rows)
        ik_actual_hz = float(payload.get("ik_actual_hz") or 0.0)
        self.panel._labels["ik_actual_hz_value"].text = f"{ik_actual_hz:.1f}"
        jacobian_mode = str(payload.get("ik_jacobian_mode") or self.panel._selected_ik_jacobian_mode() or "-")
        self.panel._labels["ik_status"].text = (
            f"joints={len(names)} jacobian={jacobian_mode}. Axis and mode changes save to main config and apply live."
        )
