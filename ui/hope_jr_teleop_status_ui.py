#!/usr/bin/env python3

import time
from typing import Any


class HopeJrTeleopStatusUi:
    def __init__(self, *, width: int = 480, height: int = 280):
        self.width = width
        self.height = height
        self._window = None
        self._labels: dict[str, Any] = {}

    def _ensure_window(self) -> None:
        if self._window is not None:
            return
        try:
            import omni.ui as ui
        except ImportError:
            return

        self._window = ui.Window("Hope Jr Teleop", width=self.width, height=self.height)
        with self._window.frame:
            with ui.VStack(spacing=4):
                with ui.VGrid(column_count=4, row_height=20, column_widths=[90, 0, 90, 0], spacing=6):
                    ui.Label("status")
                    self._labels["status"] = ui.Label("-", word_wrap=True)
                    ui.Label("messages")
                    self._labels["messages"] = ui.Label("-", word_wrap=True)

                    ui.Label("anchor")
                    self._labels["anchor"] = ui.Label("-", word_wrap=True)
                    ui.Label("grip")
                    self._labels["grip"] = ui.Label("-", word_wrap=True)

                    ui.Label("trigger")
                    self._labels["trigger"] = ui.Label("-", word_wrap=True)
                    ui.Label("buttons")
                    self._labels["buttons"] = ui.Label("-", word_wrap=True)

                with ui.VGrid(column_count=2, row_height=20, column_widths=[90, 0], spacing=6):
                    ui.Label("packet")
                    self._labels["packet"] = ui.Label("-", word_wrap=True)

                    ui.Label("mapped delta")
                    self._labels["mapped_delta"] = ui.Label("-", word_wrap=True)

                with ui.VStack(spacing=2):
                    ui.Label("markers")
                    with ui.VGrid(column_count=4, row_height=18, column_widths=[18, 0, 18, 0], spacing=6):
                        with ui.ZStack(width=12, height=12):
                            ui.Rectangle(style={"background_color": 0xFFFF8000})
                        ui.Label("QuestMapped", word_wrap=True)
                        with ui.ZStack(width=12, height=12):
                            ui.Rectangle(style={"background_color": 0xFFFF0000})
                        ui.Label("SimTarget live", word_wrap=True)

                        with ui.ZStack(width=12, height=12):
                            ui.Rectangle(style={"background_color": 0xFF00FF00})
                        ui.Label("SimTarget waiting", word_wrap=True)
                        with ui.ZStack(width=12, height=12):
                            ui.Rectangle(style={"background_color": 0xFF1A80FF})
                        ui.Label("ActualEndEffector", word_wrap=True)

    def update(self, controller, debug: dict[str, Any] | None = None) -> None:
        self._ensure_window()
        if not self._labels:
            return

        debug = debug or controller.last_debug_payload or {}
        hand = controller.last_hand_state or {}
        status = debug.get("status", "idle")
        grip = float(hand.get("grip", 0.0)) if hand else 0.0
        trigger = float(hand.get("trigger", 0.0)) if hand else 0.0

        if status == "waiting_for_anchor":
            now = float(debug.get("now", time.time()))
            ready = float(debug.get("anchor_ready_time", now))
            status_line = f"waiting ({max(0.0, ready - now):.1f}s)"
        elif controller.last_packet_timestamp is not None and grip < float(controller.grip_threshold):
            status_line = "ignored"
        else:
            status_line = str(status)

        packet_age = None
        if controller.last_packet_received_at is not None:
            packet_age = max(0.0, time.time() - controller.last_packet_received_at)
        messages = (
            "receiving"
            if packet_age is not None and packet_age <= float(controller.packet_stale_timeout_s)
            else "disconnected"
        )
        anchor = "captured" if controller.quest_anchor_position is not None else "not captured"
        buttons = (
            f"A={int(bool(hand.get('a_pressed', False)))} "
            f"B={int(bool(hand.get('b_pressed', False)))} "
            f"X={int(bool(hand.get('x_pressed', False)))} "
            f"Y={int(bool(hand.get('y_pressed', False)))} "
            f"P={int(bool(hand.get('primary_button', False)))} "
            f"S={int(bool(hand.get('secondary_button', False)))}"
        )
        packet = controller.last_packet_timestamp
        mapped_delta = debug.get("mapped_delta")
        mapped_text = "-" if mapped_delta is None else ", ".join(f"{float(v):+.3f}" for v in mapped_delta)

        if packet_age is None:
            packet_text = "-"
        else:
            packet_label = packet if packet is not None else "-"
            packet_text = f"{packet_label} ({packet_age:.2f}s ago)"

        self._labels["status"].text = status_line
        self._labels["messages"].text = messages
        self._labels["anchor"].text = anchor
        self._labels["grip"].text = f"{grip:.2f}"
        self._labels["trigger"].text = f"{trigger:.2f}"
        self._labels["buttons"].text = buttons
        self._labels["packet"].text = packet_text
        self._labels["mapped_delta"].text = mapped_text
