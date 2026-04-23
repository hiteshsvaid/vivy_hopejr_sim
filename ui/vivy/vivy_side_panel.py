#!/usr/bin/env python3

from __future__ import annotations

import time
from typing import Any


class VivySidePanel:
    _TEXT_NEUTRAL = 0xFFB8B8B8
    _TEXT_GOOD = 0xFF67C26F
    _TEXT_BAD = 0xFFD96C6C
    _TEXT_WARN = 0xFFE0BF66
    _BUS_STALE_EVENT_DELAY_S = 2.0

    def __init__(self, *, width: int = 460, height: int = 540):
        self.width = width
        self.height = height
        self._window = None
        self._labels: dict[str, Any] = {}
        self._docked = False
        self._event_rows = 12
        self._event_history_limit = 80
        self._event_history: list[dict[str, str]] = []
        self._last_event_state: dict[str, Any] | None = None
        self._bus_stale_since_monotonic: float | None = None
        self._bus_stale_event_logged = False

    def _dock_window(self, ui_module: Any) -> None:
        if self._window is None or self._docked:
            return
        try:
            workspace = ui_module.Workspace
            right_target = workspace.get_window("Hope Jr Side") or workspace.get_window("Stage")
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

                    ui.Label("QUEST", style=section_style)
                    with ui.VGrid(column_count=2, column_widths=[110, 0], row_height=22):
                        ui.Label("position", style=value_style)
                        self._labels["quest_pos"] = ui.Label("-", style=value_style)
                        ui.Label("grip / trigger", style=value_style)
                        self._labels["quest_grip_trigger"] = ui.Label("-", style=value_style)
                        ui.Label("thumbstick", style=value_style)
                        self._labels["quest_thumbstick"] = ui.Label("-", style=value_style)

                    ui.Label("STATE", style=section_style)
                    with ui.VGrid(column_count=2, column_widths=[110, 0], row_height=22):
                        ui.Label("enabled / clutch", style=value_style)
                        self._labels["state_enabled_clutch"] = ui.Label("-", style=value_style)
                        ui.Label("startup / anchor", style=value_style)
                        self._labels["state_startup_anchor"] = ui.Label("-", style=value_style)
                        ui.Label("bus", style=value_style)
                        self._labels["state_bus"] = ui.Label("-", style=value_style)
                        ui.Label("teleop rate", style=value_style)
                        self._labels["state_teleop_rate"] = ui.Label("-", style=value_style)
                        ui.Label("bus rate", style=value_style)
                        self._labels["state_bus_rate"] = ui.Label("-", style=value_style)

                    ui.Label("DELTA", style=section_style)
                    with ui.VGrid(column_count=2, column_widths=[110, 0], row_height=22):
                        ui.Label("quest", style=value_style)
                        self._labels["delta_quest"] = ui.Label("-", style=value_style)
                        ui.Label("world", style=value_style)
                        self._labels["delta_world"] = ui.Label("-", style=value_style)

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
                    with ui.VGrid(column_count=2, column_widths=[90, 0], row_height=20):
                        ui.Label("time", style=header_style)
                        ui.Label("event", style=header_style)
                    with ui.ScrollingFrame(height=170):
                        with ui.VStack(spacing=2, height=0):
                            for index in range(self._event_rows):
                                with ui.VGrid(column_count=2, column_widths=[90, 0], row_height=20):
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
    def _fmt_event_time(timestamp: float | None) -> str:
        if timestamp is None:
            return "-"
        try:
            return time.strftime("%H:%M:%S", time.localtime(float(timestamp)))
        except Exception:
            return "-"

    def _push_event(self, message: str, *, level: str = "info", timestamp: float | None = None) -> None:
        cleaned = str(message).strip()
        if not cleaned:
            return
        event = {
            "time": self._fmt_event_time(timestamp),
            "message": cleaned,
            "level": level,
        }
        if self._event_history and self._event_history[0]["message"] == event["message"]:
            self._event_history[0] = event
        else:
            self._event_history.insert(0, event)
        del self._event_history[self._event_history_limit :]

    def _update_event_history(self, payload: dict[str, Any], *, bus_live: bool, bus_status: str) -> None:
        timestamp = payload.get("timestamp")
        now_monotonic = time.monotonic()
        waiting_for_anchor = bool(payload.get("waiting_for_anchor", True))
        anchor_captured = payload.get("quest_anchor_position") is not None
        freeze_active = bool(payload.get("freeze_active", False))
        freeze_joint_name = str(payload.get("freeze_joint_name") or "").strip() or None
        follow_target_enabled = bool(payload.get("follow_target_enabled", False))
        hand = payload.get("hand_state") or {}
        grip = float(hand.get("grip", 0.0)) if isinstance(hand, dict) else 0.0
        recording_status = str(payload.get("recording_status") or "").strip() or None
        recording_name = str(payload.get("recording_name") or "").strip() or None
        state = {
            "waiting_for_anchor": waiting_for_anchor,
            "anchor_captured": anchor_captured,
            "freeze_active": freeze_active,
            "freeze_joint_name": freeze_joint_name,
            "follow_target_enabled": follow_target_enabled,
            "bus_live": bus_live,
            "bus_status": bus_status,
            "grip_active": grip >= 0.25,
            "recording_status": recording_status,
            "recording_name": recording_name,
        }
        previous = self._last_event_state
        if previous is None:
            if bus_live:
                self._bus_stale_since_monotonic = None
                self._bus_stale_event_logged = False
            else:
                self._bus_stale_since_monotonic = now_monotonic
                self._bus_stale_event_logged = False
            self._push_event(
                "Startup neutral" if waiting_for_anchor else "Startup live",
                level="info",
                timestamp=timestamp,
            )
        else:
            if waiting_for_anchor and not bool(previous.get("waiting_for_anchor", True)):
                self._push_event("Going to neutral", level="warn", timestamp=timestamp)
            elif not waiting_for_anchor and bool(previous.get("waiting_for_anchor", True)):
                self._push_event("Anchor captured", level="good", timestamp=timestamp)
            elif anchor_captured and not bool(previous.get("anchor_captured", False)):
                self._push_event("Anchor captured", level="good", timestamp=timestamp)

            if freeze_active and not bool(previous.get("freeze_active", False)):
                joint_text = "" if freeze_joint_name is None else f" on {freeze_joint_name}"
                self._push_event(f"Limit freeze engaged{joint_text}", level="bad", timestamp=timestamp)
            elif not freeze_active and bool(previous.get("freeze_active", False)):
                self._push_event("Limit freeze cleared", level="good", timestamp=timestamp)

            if follow_target_enabled and not bool(previous.get("follow_target_enabled", False)):
                self._push_event("Motion tracking on", level="good", timestamp=timestamp)
            elif not follow_target_enabled and bool(previous.get("follow_target_enabled", False)):
                self._push_event("Motion tracking off", level="warn", timestamp=timestamp)

            stale_was_logged = self._bus_stale_event_logged
            if bus_live:
                self._bus_stale_since_monotonic = None
                self._bus_stale_event_logged = False
            elif bool(previous.get("bus_live", False)) and self._bus_stale_since_monotonic is None:
                self._bus_stale_since_monotonic = now_monotonic

            if bus_live and not bool(previous.get("bus_live", False)) and stale_was_logged:
                self._push_event(f"Bus live ({bus_status})", level="good", timestamp=timestamp)
            elif (
                not bus_live
                and self._bus_stale_since_monotonic is not None
                and not self._bus_stale_event_logged
                and (now_monotonic - self._bus_stale_since_monotonic) >= self._BUS_STALE_EVENT_DELAY_S
            ):
                self._push_event(f"Bus stale ({bus_status})", level="warn", timestamp=timestamp)
                self._bus_stale_event_logged = True

            if recording_status != previous.get("recording_status"):
                if recording_status == "recording":
                    name_text = "" if not recording_name else f" {recording_name}"
                    self._push_event(f"Recording started{name_text}", level="good", timestamp=timestamp)
                elif recording_status == "recording_ended":
                    name_text = "" if not recording_name else f" {recording_name}"
                    self._push_event(f"Recording stopped{name_text}", level="warn", timestamp=timestamp)
                elif recording_status == "waiting_for_a" and recording_name:
                    self._push_event(f"Recording armed {recording_name}", level="info", timestamp=timestamp)
        self._last_event_state = state

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
        grip = float(hand.get("grip", 0.0)) if isinstance(hand, dict) else 0.0
        trigger = float(hand.get("trigger", 0.0)) if isinstance(hand, dict) else 0.0
        thumbstick = hand.get("thumbstick") if isinstance(hand, dict) else None
        enabled = bool(hand.get("enabled", True)) if isinstance(hand, dict) else False
        clutch = bool(hand.get("clutch", False)) if isinstance(hand, dict) else False
        waiting_for_anchor = bool(payload.get("waiting_for_anchor", True))
        startup = "neutral" if waiting_for_anchor else "live"
        anchor = "captured" if payload.get("quest_anchor_position") is not None else "pending"
        frozen = "yes" if payload.get("freeze_active") else "no"

        self._labels["quest_pos"].text = self._fmt_vec3(hand.get("position"))
        self._labels["quest_grip_trigger"].text = f"grip={grip:.2f}  trigger={trigger:.2f}"
        thumbstick_label = self._labels.get("quest_thumbstick")
        if thumbstick_label is not None:
            if isinstance(thumbstick, (list, tuple)) and len(thumbstick) == 2:
                try:
                    thumbstick_label.text = f"x={float(thumbstick[0]):+.2f}  y={float(thumbstick[1]):+.2f}"
                except Exception:
                    thumbstick_label.text = "-"
            else:
                thumbstick_label.text = "-"
        self._labels["state_enabled_clutch"].text = f"enabled={enabled}  clutch={clutch}"
        self._labels["state_startup_anchor"].text = f"startup={startup}  anchor={anchor}  frozen={frozen}"
        bus_live = bool(payload.get("real_feedback_live", False))
        bus_status = str(payload.get("real_feedback_status") or ("live" if bus_live else "stale"))
        self._update_event_history(payload, bus_live=bus_live, bus_status=bus_status)
        self._labels["state_bus"].text = bus_status
        try:
            self._labels["state_bus"].style = {"color": self._TEXT_GOOD if bus_live else self._TEXT_BAD, "font_size": 13}
        except Exception:
            pass
        teleop_hz = payload.get("teleop_hz")
        bus_hz = payload.get("bus_hz")
        self._labels["state_teleop_rate"].text = "-" if teleop_hz is None else f"{float(teleop_hz):.1f} Hz"
        self._labels["state_bus_rate"].text = "-" if bus_hz is None else f"{float(bus_hz):.1f} Hz"
        self._labels["delta_quest"].text = self._fmt_vec3(payload.get("quest_delta"))
        self._labels["delta_world"].text = self._fmt_vec3(payload.get("position_delta_world"))

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
