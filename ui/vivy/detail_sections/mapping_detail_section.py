from __future__ import annotations

from typing import Any


class MappingDetailSection:
    def __init__(self, panel: Any):
        self.panel = panel

    def build(self, ui: Any, parent: Any, header_style: dict[str, Any], value_style: dict[str, Any]) -> None:
        with ui.VStack(spacing=4) as axis_editor:
            self.panel._labels["axis_editor"] = axis_editor
            ui.Label("Axis / Sign Remap", style=header_style)
            with ui.HStack(height=26, spacing=6):
                ui.Label("axes", width=48, style=value_style)
                axis_field = ui.StringField(width=80)
                self.panel._labels["axis_axes_model"] = axis_field.model
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
            with ui.HStack(height=26, spacing=6):
                ui.Label("target_max_delta", width=120, style=value_style)
                target_delta_field = ui.StringField(width=80)
                self.panel._labels["target_max_delta_model"] = target_delta_field.model
                ui.Label("m/tick", width=48, style=value_style)
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
