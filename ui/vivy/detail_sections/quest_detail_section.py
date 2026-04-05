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
            self.panel._labels["quest_status"] = ui.Label("", style=value_style, word_wrap=True)

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
