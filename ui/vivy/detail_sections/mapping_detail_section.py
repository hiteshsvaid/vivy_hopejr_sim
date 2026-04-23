from __future__ import annotations

from typing import Any


class MappingDetailSection:
    def __init__(self, panel: Any):
        self.panel = panel

    def build(self, ui: Any, parent: Any, header_style: dict[str, Any], value_style: dict[str, Any]) -> None:
        def add_row(label: str, model_key: str, *, width: int = 90, units: str | None = None) -> None:
            with ui.HStack(height=26, spacing=6):
                ui.Label(label, width=170, style=value_style)
                field = ui.StringField(width=width)
                self.panel._labels[model_key] = field.model
                if units:
                    ui.Label(units, width=64, style=value_style)

        with ui.VStack(spacing=4) as axis_editor:
            self.panel._labels["axis_editor"] = axis_editor
            ui.Label("Position Mapping", style=header_style)
            with ui.HStack(height=26, spacing=6):
                ui.Label("axes", width=48, style=value_style)
                axis_field = ui.StringField(width=80)
                self.panel._labels["axis_axes_model"] = axis_field.model
            ui.Label(
                "Reorders Quest controller translation axes before IK. Use permutations like xyz or zxy.",
                style=value_style,
                word_wrap=True,
            )
            with ui.HStack(height=26, spacing=6):
                ui.Label("sign x", width=48, style=value_style)
                sign_x = ui.StringField(width=60)
                self.panel._labels["axis_sign_x_model"] = sign_x.model
                ui.Label("sign y", width=48, style=value_style)
                sign_y = ui.StringField(width=60)
                self.panel._labels["axis_sign_y_model"] = sign_y.model
                ui.Label("sign z", width=48, style=value_style)
                sign_z = ui.StringField(width=60)
                self.panel._labels["axis_sign_z_model"] = sign_z.model
            ui.Label(
                "Flips the remapped translation axes. Use 1.0 for normal and -1.0 to invert.",
                style=value_style,
                word_wrap=True,
            )
            add_row("position_scale", "position_scale_model")
            ui.Label(
                "Scales Quest translation before IK. Larger values make controller motion move the target more.",
                style=value_style,
                word_wrap=True,
            )
            add_row("target_max_delta", "target_max_delta_model", units="m/tick")
            ui.Label(
                "Limits target-position change per control tick before IK sees it.",
                style=value_style,
                word_wrap=True,
            )
            self.panel._labels["axis_save_button"] = ui.Button(
                "Save Mapping",
                height=28,
                clicked_fn=lambda: self.panel._save_axis_remap(),
            )
            self.panel._labels["axis_status"] = ui.Label("", style=value_style, word_wrap=True)

    def load_from_config(self) -> None:
        self.panel._load_axis_remap_fields_from_config()
