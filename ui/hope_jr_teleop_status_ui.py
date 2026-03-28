#!/usr/bin/env python3

import time
from typing import Any


class HopeJrTeleopStatusUi:
    def __init__(self, *, width: int = 620, height: int = 420):
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

                    ui.Label("profile")
                    self._labels["profile"] = ui.Label("-", word_wrap=True)
                    ui.Label("err score")
                    self._labels["error_score"] = ui.Label("-", word_wrap=True)

                with ui.VGrid(column_count=2, row_height=20, column_widths=[90, 0], spacing=6):
                    ui.Label("packet")
                    self._labels["packet"] = ui.Label("-", word_wrap=True)

                    ui.Label("mapped delta")
                    self._labels["mapped_delta"] = ui.Label("-", word_wrap=True)

                with ui.VStack(spacing=2):
                    with ui.VGrid(column_count=4, row_height=20, column_widths=[180, 60, 60, 50], spacing=6):
                        self._labels["joint_header_name"] = ui.Label("joint")
                        self._labels["joint_header_start"] = ui.Label("start")
                        self._labels["joint_header_cur"] = ui.Label("cur")
                        self._labels["joint_header_wt"] = ui.Label("wt")
                        for idx in range(7):
                            self._labels[f"joint_name_{idx}"] = ui.Label("-")
                            self._labels[f"joint_start_{idx}"] = ui.Label("-")
                            self._labels[f"joint_cur_{idx}"] = ui.Label("-")
                            self._labels[f"joint_wt_{idx}"] = ui.Label("-")

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

        packet_age = None
        if controller.last_packet_received_at is not None:
            packet_age = max(0.0, time.time() - controller.last_packet_received_at)
        sim_playing = True
        try:
            import omni.timeline
            timeline = omni.timeline.get_timeline_interface()
            sim_playing = bool(timeline.is_playing()) if timeline is not None else True
        except Exception:
            sim_playing = True

        if not sim_playing:
            status_line = "paused"
        elif status == "waiting_for_anchor":
            now = float(debug.get("now", time.time()))
            ready = float(debug.get("anchor_ready_time", now))
            status_line = f"waiting ({max(0.0, ready - now):.1f}s)"
        elif controller.last_packet_timestamp is not None and grip < float(controller.teleop_mapper.grip_threshold):
            status_line = "ignored"
        else:
            status_line = str(status)
        messages = (
            "paused"
            if not sim_playing
            else (
                "receiving"
                if packet_age is not None and packet_age <= float(controller.packet_stale_timeout_s)
                else "disconnected"
            )
        )
        anchor = "captured" if controller.teleop_mapper.quest_anchor_position is not None else "not captured"
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
        result = debug.get("result") or {}
        stage_joint_positions_deg = result.get("stage_joint_positions_deg") or debug.get("stage_joint_positions_deg")
        stage_start_joint_positions_deg = result.get("stage_start_joint_positions_deg") or debug.get("stage_start_joint_positions_deg")
        joint_names = result.get("joint_names") or []
        joint_weights = result.get("stage_dls_joint_weights") or []
        joint_rows = []
        if stage_joint_positions_deg:
            for idx, angle in enumerate(stage_joint_positions_deg):
                name = joint_names[idx] if idx < len(joint_names) else f"joint_{idx}"
                start_angle = stage_start_joint_positions_deg[idx] if stage_start_joint_positions_deg and idx < len(stage_start_joint_positions_deg) else angle
                weight = joint_weights[idx] if idx < len(joint_weights) else 0.0
                joint_rows.append((name, f"{float(start_angle):+.1f}", f"{float(angle):+.1f}", f"{float(weight):.2f}"))
        stage_profile = debug.get("stage_weight_profile") or result.get("stage_weight_profile") or getattr(controller, "stage_weight_profile", "-")
        stage_error_score = debug.get("stage_error_score") or (debug.get("result") or {}).get("stage_error_score")
        if stage_error_score is None:
            error_text = "-"
        else:
            error_text = (
                f"mean={float(stage_error_score.get('mean_error_norm_m', 0.0)):.4f} "
                f"latest={float(stage_error_score.get('latest_error_norm_m', 0.0)):.4f} "
                f"n={int(stage_error_score.get('window_size', 0))}"
            )

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
        self._labels["profile"].text = str(stage_profile)
        self._labels["error_score"].text = error_text
        self._labels["packet"].text = packet_text
        self._labels["mapped_delta"].text = mapped_text
        for idx in range(7):
            if idx < len(joint_rows):
                name, start_text, cur_text, wt_text = joint_rows[idx]
            else:
                name, start_text, cur_text, wt_text = ("-", "-", "-", "-")
            self._labels[f"joint_name_{idx}"].text = name
            self._labels[f"joint_start_{idx}"].text = start_text
            self._labels[f"joint_cur_{idx}"].text = cur_text
            self._labels[f"joint_wt_{idx}"].text = wt_text
