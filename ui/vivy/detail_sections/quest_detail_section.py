from __future__ import annotations

from typing import Any


class QuestDetailSection:
    def __init__(self, panel: Any):
        self.panel = panel

    def build(self, ui: Any, parent: Any, header_style: dict[str, Any], value_style: dict[str, Any]) -> None:
        with ui.VStack(spacing=4) as quest_editor:
            self.panel._labels["quest_editor"] = quest_editor
            ui.Label("Quest Files", style=header_style)
            with ui.HStack(height=26, spacing=6) as replay_container:
                self.panel._quest_replay_container = replay_container
                self.panel._refresh_replay_dropdown()
            self.panel._labels["quest_save_button"] = ui.Button(
                "Save Replay Name",
                height=28,
                clicked_fn=lambda: self.panel._save_replay_name(),
            )
            self.panel._labels["quest_delete_button"] = ui.Button(
                "Delete Replay",
                height=28,
                clicked_fn=lambda: self.panel._delete_selected_replay(),
            )
            with ui.HStack(height=26, spacing=6):
                ui.Label("record", width=48, style=value_style)
                record_field = ui.StringField(width=220)
                self.panel._labels["quest_record_model"] = record_field.model
            self.panel._labels["quest_record_button"] = ui.Button(
                "Save Record Name",
                height=28,
                clicked_fn=lambda: self.panel._save_record_name(),
            )
            ui.Spacer(height=4)
            ui.Label("Camera Stream", style=header_style)
            with ui.VStack(spacing=2):
                with ui.HStack(height=26, spacing=6):
                    ui.Label("publish_hz", width=120, style=value_style)
                    publish_hz_field = ui.StringField(width=90)
                    self.panel._labels["quest_camera_publish_hz_model"] = publish_hz_field.model
                ui.Label(
                    "Camera frame send rate for the active publisher. Lower values reduce bandwidth and latency pressure.",
                    style=value_style,
                    word_wrap=True,
                )
                with ui.HStack(height=26, spacing=6):
                    ui.Label("jpeg_quality", width=120, style=value_style)
                    jpeg_quality_field = ui.StringField(width=90)
                    self.panel._labels["quest_camera_jpeg_quality_model"] = jpeg_quality_field.model
                ui.Label(
                    "JPEG quality used to compress the camera stream before sending it to Quest.",
                    style=value_style,
                    word_wrap=True,
                )
                with ui.HStack(height=26, spacing=6):
                    ui.Label("resolution", width=120, style=value_style)
                    resolution_field = ui.StringField(width=90)
                    self.panel._labels["quest_camera_resolution_model"] = resolution_field.model
                ui.Label(
                    "Camera frame resolution like 320x240. Lower values reduce bandwidth and can reduce latency.",
                    style=value_style,
                    word_wrap=True,
                )
                with ui.HStack(height=26, spacing=6):
                    ui.Label("camera diagnostics", width=120, style=value_style)
                    diagnostics_checkbox = ui.CheckBox(width=24)
                    self.panel._labels["quest_camera_diagnostics_model"] = diagnostics_checkbox.model
                ui.Label(
                    "When enabled, the camera publisher prints timing diagnostics to the console.",
                    style=value_style,
                    word_wrap=True,
                )
            self.panel._labels["quest_camera_save_button"] = ui.Button(
                "Save Camera Settings",
                height=28,
                clicked_fn=lambda: self.panel._save_camera_stream_settings(),
            )
            self.panel._labels["quest_status"] = ui.Label("", style=value_style, word_wrap=True)

    def load_from_config(self) -> None:
        config = self.panel._read_vivy_config()
        controller_defaults = dict(config.get("controller_defaults") or {})
        camera_config = dict(controller_defaults.get("camera") or {})
        stream_config = dict(camera_config.get("stream") or {})
        try:
            self.panel._labels["quest_camera_publish_hz_model"].set_value(
                str(stream_config.get("publish_hz", 5.0))
            )
            self.panel._labels["quest_camera_jpeg_quality_model"].set_value(
                str(stream_config.get("jpeg_quality", 60))
            )
            resolution = stream_config.get("resolution", [320, 240])
            if isinstance(resolution, (list, tuple)) and len(resolution) == 2:
                resolution_text = f"{int(float(resolution[0]))}x{int(float(resolution[1]))}"
            else:
                resolution_text = "320x240"
            self.panel._labels["quest_camera_resolution_model"].set_value(resolution_text)
            self.panel._labels["quest_camera_diagnostics_model"].set_value(
                bool(stream_config.get("show_diagnostics", False))
            )
        except Exception:
            pass

    def update(self, replay_name: str, record_name: str, recording_name: str, recording_status: str, recording_packet_count: int) -> None:
        names = self.panel._list_recording_names()
        if names != self.panel._quest_recording_names:
            self.panel._refresh_replay_dropdown(replay_name if replay_name != "-" else None)
            names = self.panel._quest_recording_names
        elif names and replay_name != self.panel._last_saved_replay_name:
            selected_index = names.index(replay_name) if replay_name in names else 0
            self.panel._labels["quest_replay_combo_model"].get_item_value_model().set_value(selected_index)
        if record_name != self.panel._last_saved_record_name:
            self.panel._labels["quest_record_model"].set_value(record_name if record_name != "-" else "")
        self.panel._last_saved_replay_name = replay_name
        self.panel._last_saved_record_name = record_name
        if recording_status == "waiting_for_a":
            status_text = f"Recording armed for {recording_name}. Waiting for A."
        elif recording_status == "recording":
            status_text = f"Recording {recording_name}. packets={recording_packet_count}"
        elif recording_status == "recording_ended":
            status_text = f"Recording ended for {recording_name}. packets={recording_packet_count}"
        else:
            status_text = "Set replay or record names here."
        self.panel._labels["quest_status"].text = status_text
