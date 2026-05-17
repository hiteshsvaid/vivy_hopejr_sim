#!/usr/bin/env python3

from __future__ import annotations

import time
from typing import Any


class VivySidePanel:
    _TEXT_NEUTRAL = 0xFFB8B8B8
    _TEXT_GOOD = 0xFF67C26F
    _TEXT_BAD = 0xFFD96C6C
    _TEXT_WARN = 0xFFE0BF66

    def __init__(self, *, width: int = 460, height: int = 540):
        self.width = width
        self.height = height
        self._window = None
        self._labels: dict[str, Any] = {}
        self._docked = False
        self._event_rows = 12
        self._event_history_limit = 80
        self._event_history: list[dict[str, Any]] = []
        self._event_batch: list[dict[str, Any]] | None = None
        self._event_sequence = 0

    def _dock_window(self, ui_module: Any) -> None:
        if self._window is None or self._docked:
            return
        try:
            workspace = ui_module.Workspace
            right_target = workspace.get_window("Vivy Side") or workspace.get_window("Stage")
            if right_target is not None:
                ui_module.Workspace.show_window("Vivy Quest", True)
                self._window.dock_in(right_target, ui_module.DockPosition.SAME)
                try:
                    self._window.focus()
                except Exception:
                    pass
                self._docked = True
        except Exception:
            pass

    def _ensure_window(self) -> None:
        if self._window is not None:
            return
        try:
            import omni.ui as ui
        except ImportError:
            return

        self._window = ui.Window("Vivy Quest", width=self.width, height=self.height)
        try:
            self._window.deferred_dock_in("Stage")
        except Exception:
            pass
        try:
            self._window.focus()
        except Exception:
            pass
        with self._window.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=6, height=0):
                    section_style = {"color": 0xFFD8D8D8, "font_size": 13}
                    value_style = {"color": self._TEXT_NEUTRAL, "font_size": 13}
                    joint_value_style = {"color": self._TEXT_NEUTRAL, "font_size": 12}
                    header_style = {"color": 0xFFD8D8D8, "font_size": 12}
                    joint_name_style = {"color": 0xFFD8D8D8, "font_size": 12}

                    ui.Label("BUS", style=section_style)
                    with ui.VGrid(column_count=2, column_widths=[110, 0], row_height=22):
                        ui.Label("bus", style=value_style)
                        self._labels["state_bus"] = ui.Label("-", style=value_style)
                        ui.Label("teleop rate", style=value_style)
                        self._labels["state_teleop_rate"] = ui.Label("-", style=value_style)
                        ui.Label("bus rate", style=value_style)
                        self._labels["state_bus_rate"] = ui.Label("-", style=value_style)

                    ui.Spacer(height=4)
                    ui.Label("JOINTS", style=section_style)
                    with ui.VGrid(column_count=7, column_widths=[45, 45, 65, 75, 55, 45, 45], row_height=20):
                        ui.Label("MIN", style=header_style)
                        ui.Label("D_DEG", style=header_style)
                        ui.Label("IK / ACT", style=header_style)
                        ui.Label("SERVO / ACT", style=header_style)
                        ui.Label("ERR", style=header_style)
                        ui.Label("MAX", style=header_style)
                        ui.Label("MODE", style=header_style)
                    for index in range(7):
                        with ui.VStack(spacing=2):
                            self._labels[f"joint_name_{index}"] = ui.Label("", style=joint_name_style)
                            with ui.VGrid(column_count=7, column_widths=[45, 45, 65, 75, 55, 45, 45], row_height=20):
                                self._labels[f"joint_min_deg_{index}"] = ui.Label("", style=joint_value_style)
                                self._labels[f"joint_delta_deg_{index}"] = ui.Label("", style=joint_value_style)
                                with ui.VStack(spacing=0):
                                    self._labels[f"joint_target_deg_{index}"] = ui.Label("", style=joint_value_style)
                                    self._labels[f"joint_actual_deg_{index}"] = ui.Label("", style=joint_value_style)
                                with ui.VStack(spacing=0):
                                    self._labels[f"joint_raw_cmd_{index}"] = ui.Label("", style=joint_value_style)
                                    self._labels[f"joint_actual_raw_{index}"] = ui.Label("", style=joint_value_style)
                                self._labels[f"joint_error_deg_{index}"] = ui.Label("", style=joint_value_style)
                                self._labels[f"joint_max_deg_{index}"] = ui.Label("", style=joint_value_style)
                                self._labels[f"joint_mode_{index}"] = ui.Label("", style=joint_value_style)

                    ui.Spacer(height=6)
                    ui.Label("EVENTS", style=section_style)
                    with ui.VGrid(column_count=2, column_widths=[145, 0], row_height=20):
                        ui.Label("time", style=header_style)
                        ui.Label("event", style=header_style)
                    with ui.ScrollingFrame(height=170):
                        with ui.VStack(spacing=2, height=0):
                            for index in range(self._event_rows):
                                with ui.VGrid(column_count=2, column_widths=[145, 0], row_height=20):
                                    self._labels[f"event_time_{index}"] = ui.Label("", style=joint_value_style)
                                    self._labels[f"event_message_{index}"] = ui.Label("", style=joint_value_style)
        self._dock_window(ui)

    @staticmethod
    def _fmt_vec3(value: Any) -> str:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            return "-"
        try:
            return f"({float(value[0]):+.3f}, {float(value[1]):+.3f}, {float(value[2]):+.3f})"
        except Exception:
            return "-"

    @staticmethod
    def _fmt_event_time(timestamp: float | None, timestamp_ns: int | None, sequence: int) -> str:
        parsed_timestamp_ns = None
        if timestamp_ns is not None:
            try:
                parsed_timestamp_ns = int(timestamp_ns)
            except Exception:
                parsed_timestamp_ns = None
        if parsed_timestamp_ns is None:
            if timestamp is None:
                parsed_timestamp_ns = time.time_ns()
            else:
                try:
                    parsed_timestamp_ns = int(float(timestamp) * 1_000_000_000)
                except Exception:
                    parsed_timestamp_ns = time.time_ns()
        seconds = parsed_timestamp_ns / 1_000_000_000.0
        nanoseconds = parsed_timestamp_ns % 1_000_000_000
        return f"{time.strftime('%H:%M:%S', time.localtime(seconds))}.{nanoseconds:09d}.{sequence % 1000:03d}"

    def _push_event(
        self,
        message: str,
        *,
        level: str = "info",
        timestamp: float | None = None,
        timestamp_ns: int | None = None,
    ) -> None:
        cleaned = str(message).strip()
        if not cleaned:
            return
        self._event_sequence += 1
        event = {
            "time": self._fmt_event_time(timestamp, timestamp_ns, self._event_sequence),
            "message": cleaned,
            "level": level,
            "sequence": self._event_sequence,
        }
        if self._event_batch is not None:
            if self._event_batch and self._event_batch[-1]["message"] == event["message"]:
                self._event_batch[-1] = event
            else:
                self._event_batch.append(event)
            return
        if self._event_history and self._event_history[0]["message"] == event["message"]:
            self._event_history[0] = event
        else:
            self._event_history.insert(0, event)
        self._event_history.sort(key=lambda row: int(row.get("sequence", 0)), reverse=True)
        del self._event_history[self._event_history_limit :]

    def _begin_event_batch(self) -> None:
        self._event_batch = []

    def _flush_event_batch(self) -> None:
        batch = self._event_batch or []
        self._event_batch = None
        if not batch:
            return
        if self._event_history and self._event_history[0]["message"] == batch[-1]["message"]:
            self._event_history[0] = batch[-1]
            batch = batch[:-1]
        self._event_history = batch + self._event_history
        self._event_history.sort(key=lambda row: int(row.get("sequence", 0)), reverse=True)
        del self._event_history[self._event_history_limit :]

    def _update_event_history(
        self,
        payload: dict[str, Any],
    ) -> None:
        timestamp = payload.get("timestamp")
        timestamp_ns = payload.get("timestamp_ns")
        event_messages = payload.get("event_messages")
        if isinstance(event_messages, list):
            source_events = [event for event in event_messages if isinstance(event, dict)]
        else:
            source_events = []
        if not source_events and str(payload.get("event_message") or "").strip():
            source_events = [payload]
        for source_event in source_events:
            event_message = str(source_event.get("event_message") or "").strip()
            if not event_message:
                continue
            self._push_event(
                event_message,
                level=str(source_event.get("event_level") or "info"),
                timestamp=source_event.get("timestamp", timestamp),
                timestamp_ns=source_event.get("timestamp_ns", timestamp_ns),
            )

    def _event_style(self, level: str) -> dict[str, int]:
        color = self._TEXT_NEUTRAL
        if level == "good":
            color = self._TEXT_GOOD
        elif level == "warn":
            color = self._TEXT_WARN
        elif level == "bad":
            color = self._TEXT_BAD
        return {"color": color, "font_size": 12}

    def update(self, payload: dict[str, Any] | None = None) -> None:
        self._ensure_window()
        if not self._labels:
            return
        try:
            import omni.ui as ui
            self._dock_window(ui)
        except Exception:
            pass

        payload = payload or {}
        hand = payload.get("hand_state") or {}
        thumbstick = hand.get("thumbstick") if isinstance(hand, dict) else None
        thumbstick_click = bool(hand.get("thumbstick_click", False)) if isinstance(hand, dict) else False
        trigger_button = bool(hand.get("trigger_button", False)) if isinstance(hand, dict) else False
        grip_button = bool(hand.get("grip_button", False)) if isinstance(hand, dict) else False

        thumbstick_label = self._labels.get("quest_thumbstick")
        if thumbstick_label is not None:
            if isinstance(thumbstick, (list, tuple)) and len(thumbstick) == 2:
                try:
                    thumbstick_label.text = f"x={float(thumbstick[0]):+.2f}  y={float(thumbstick[1]):+.2f}"
                except Exception:
                    thumbstick_label.text = "-"
            else:
                thumbstick_label.text = "-"
        bus_live = bool(payload.get("real_feedback_live", False))
        bus_status = str(payload.get("real_feedback_status") or ("live" if bus_live else "stale"))
        self._begin_event_batch()
        try:
            self._update_event_history(payload)
        finally:
            self._flush_event_batch()
        self._labels["state_bus"].text = bus_status
        try:
            self._labels["state_bus"].style = {"color": self._TEXT_GOOD if bus_live else self._TEXT_BAD, "font_size": 13}
        except Exception:
            pass
        teleop_hz = payload.get("teleop_hz")
        bus_hz = payload.get("bus_hz")
        self._labels["state_teleop_rate"].text = "-" if teleop_hz is None else f"{float(teleop_hz):.1f} Hz"
        self._labels["state_bus_rate"].text = "-" if bus_hz is None else f"{float(bus_hz):.1f} Hz"
        rows = payload.get("joint_display_rows") or []
        for index in range(7):
            self._labels[f"joint_name_{index}"].text = ""
            self._labels[f"joint_min_deg_{index}"].text = ""
            self._labels[f"joint_delta_deg_{index}"].text = ""
            self._labels[f"joint_target_deg_{index}"].text = ""
            self._labels[f"joint_raw_cmd_{index}"].text = ""
            self._labels[f"joint_actual_deg_{index}"].text = ""
            self._labels[f"joint_actual_raw_{index}"].text = ""
            self._labels[f"joint_error_deg_{index}"].text = ""
            self._labels[f"joint_max_deg_{index}"].text = ""
            self._labels[f"joint_mode_{index}"].text = ""
            if index < len(rows):
                row = rows[index]
                self._labels[f"joint_name_{index}"].text = str(row.get("joint") or "")
                self._labels[f"joint_min_deg_{index}"].text = str(row.get("min_deg") or "").strip()
                self._labels[f"joint_delta_deg_{index}"].text = str(row.get("delta_deg") or "").strip()
                self._labels[f"joint_target_deg_{index}"].text = str(row.get("target_deg") or "").strip()
                self._labels[f"joint_raw_cmd_{index}"].text = str(row.get("raw_cmd") or "").strip()
                self._labels[f"joint_actual_deg_{index}"].text = str(row.get("actual_deg") or "").strip()
                self._labels[f"joint_actual_raw_{index}"].text = str(row.get("actual_raw") or "").strip()
                self._labels[f"joint_error_deg_{index}"].text = str(row.get("error_deg") or "").strip()
                self._labels[f"joint_max_deg_{index}"].text = str(row.get("max_deg") or "").strip()
                self._labels[f"joint_mode_{index}"].text = str(row.get("mode") or "")

        for index in range(self._event_rows):
            self._labels[f"event_time_{index}"].text = ""
            self._labels[f"event_message_{index}"].text = ""
            try:
                neutral_style = self._event_style("info")
                self._labels[f"event_time_{index}"].style = neutral_style
                self._labels[f"event_message_{index}"].style = neutral_style
            except Exception:
                pass
            if index < len(self._event_history):
                event = self._event_history[index]
                self._labels[f"event_time_{index}"].text = event["time"]
                self._labels[f"event_message_{index}"].text = event["message"]
                try:
                    style = self._event_style(event["level"])
                    self._labels[f"event_time_{index}"].style = style
                    self._labels[f"event_message_{index}"].style = style
                except Exception:
                    pass
