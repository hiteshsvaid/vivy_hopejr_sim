#!/usr/bin/env python3

import time
from typing import Any


class VivyTeleopSidePanel:
    _TEXT_NEUTRAL = 0xFFB8B8B8
    _TEXT_DIM = 0xFF8E8E8E
    _TEXT_CRITICAL = 0xFFFF5A5A

    def __init__(self, *, width: int = 720, height: int = 400):
        self.width = width
        self.height = height
        self._window = None
        self._labels: dict[str, Any] = {}

    @classmethod
    def _joint_current_color(
        cls,
        *,
        current_deg: float | None,
        target_deg: float | None,
        lower_limit_deg: float | None,
        upper_limit_deg: float | None,
        saturation_tol_deg: float = 0.5,
        push_tol_deg: float = 0.25,
    ) -> int:
        if current_deg is None or target_deg is None or lower_limit_deg is None or upper_limit_deg is None:
            return cls._TEXT_NEUTRAL
        current_deg = float(current_deg)
        target_deg = float(target_deg)
        lower_limit_deg = float(lower_limit_deg)
        upper_limit_deg = float(upper_limit_deg)
        pinned_upper = current_deg >= (upper_limit_deg - saturation_tol_deg)
        pinned_lower = current_deg <= (lower_limit_deg + saturation_tol_deg)
        pushing_upper = target_deg > max(upper_limit_deg + push_tol_deg, current_deg + push_tol_deg)
        pushing_lower = target_deg < min(lower_limit_deg - push_tol_deg, current_deg - push_tol_deg)
        if (pinned_upper and pushing_upper) or (pinned_lower and pushing_lower):
            return cls._TEXT_CRITICAL
        return cls._TEXT_NEUTRAL

    @staticmethod
    def _set_label_color(label: Any, color: int) -> None:
        try:
            label.style = {"color": color}
        except Exception:
            pass

    def _ensure_window(self) -> None:
        if self._window is not None:
            return
        try:
            import omni.ui as ui
        except ImportError:
            return

        self._window = ui.Window("Vivy Side", width=self.width, height=self.height)
        try:
            self._window.focus()
        except Exception:
            pass
        with self._window.frame:
            with ui.ScrollingFrame(
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            ):
                with ui.VStack(spacing=3):
                    label_style = {"color": self._TEXT_NEUTRAL, "font_size": 13}
                    section_style = {"color": 0xFFD8D8D8, "font_size": 13}

                    with ui.VGrid(column_count=4, column_widths=[0, 0, 0, 0], spacing=6):
                        with ui.VStack(spacing=0):
                            ui.Label("STATUS", style=label_style)
                            self._labels["status"] = ui.Label("-", word_wrap=True)
                        with ui.VStack(spacing=0):
                            ui.Label("MESSAGES", style=label_style)
                            self._labels["messages"] = ui.Label("-", word_wrap=True)
                        with ui.VStack(spacing=0):
                            ui.Label("ANCHOR", style=label_style)
                            self._labels["anchor"] = ui.Label("-", word_wrap=True)
                        with ui.VStack(spacing=0):
                            ui.Label("GRIP", style=label_style)
                            self._labels["grip"] = ui.Label("-", word_wrap=True)

                        with ui.VStack(spacing=0):
                            ui.Label("TRIGGER", style=label_style)
                            self._labels["trigger"] = ui.Label("-", word_wrap=True)
                        with ui.VStack(spacing=0):
                            ui.Label("BUTTONS", style=label_style)
                            self._labels["buttons"] = ui.Label("-", word_wrap=True)
                        with ui.VStack(spacing=0):
                            ui.Label("CONFIG", style=label_style)
                            self._labels["profile"] = ui.Label("-", word_wrap=True)
                        with ui.VStack(spacing=0):
                            ui.Label("", style=label_style)
                            self._labels["spacer_top"] = ui.Label("", word_wrap=True)

                    ui.Line(style={"color": 0x55FFFFFF})

                    with ui.HStack(height=18, spacing=6):
                        ui.Label("PACKET DATA", style=section_style, width=110)
                        self._labels["packet"] = ui.Label("-", word_wrap=False)

                    ui.Line(style={"color": 0x55FFFFFF})

                    with ui.VGrid(column_count=2, column_widths=[0, 0], spacing=6):
                        with ui.VStack(spacing=0):
                            ui.Label("MAPPED DELTA", style=section_style)
                            self._labels["mapped_delta"] = ui.Label("-", word_wrap=True)
                        with ui.VStack(spacing=0):
                            ui.Label("ERROR", style=section_style)
                            self._labels["error_score"] = ui.Label("-", word_wrap=True)

                    ui.Line(style={"color": 0x55FFFFFF})

                    with ui.VStack(spacing=2):
                        with ui.VGrid(column_count=6, row_height=18, column_widths=[56, 56, 56, 56, 56, 42], spacing=6):
                            self._labels["joint_header_start"] = ui.Label("start", style=label_style)
                            self._labels["joint_header_min"] = ui.Label("min", style=label_style)
                            self._labels["joint_header_cur"] = ui.Label("cur", style=label_style)
                            self._labels["joint_header_tgt"] = ui.Label("tgt", style=label_style)
                            self._labels["joint_header_max"] = ui.Label("max", style=label_style)
                            self._labels["joint_header_wt"] = ui.Label("wt", style=label_style)
                        for idx in range(7):
                            with ui.VStack(spacing=0):
                                self._labels[f"joint_name_{idx}"] = ui.Label("-", word_wrap=True)
                                with ui.VGrid(column_count=6, row_height=18, column_widths=[56, 56, 56, 56, 56, 42], spacing=6):
                                    self._labels[f"joint_start_{idx}"] = ui.Label("-")
                                    self._labels[f"joint_min_{idx}"] = ui.Label("-")
                                    self._labels[f"joint_cur_{idx}"] = ui.Label("-")
                                    self._labels[f"joint_tgt_{idx}"] = ui.Label("-")
                                    self._labels[f"joint_max_{idx}"] = ui.Label("-")
                                    self._labels[f"joint_wt_{idx}"] = ui.Label("-")

                    ui.Line(style={"color": 0x55FFFFFF})

                    with ui.VStack(spacing=2):
                        ui.Label("MARKERS", style=section_style)
                        with ui.VGrid(column_count=2, row_height=18, column_widths=[18, 0], spacing=6):
                            with ui.ZStack(width=12, height=12):
                                ui.Rectangle(style={"background_color": 0xFFFF8000})
                            ui.Label("RightQuestMapped", word_wrap=True)
                            with ui.ZStack(width=12, height=12):
                                ui.Rectangle(style={"background_color": 0xFFFF0000})
                            ui.Label("RightSimTarget live", word_wrap=True)
                            with ui.ZStack(width=12, height=12):
                                ui.Rectangle(style={"background_color": 0xFF66AFFF})
                            ui.Label("LeftQuestMapped", word_wrap=True)
                            with ui.ZStack(width=12, height=12):
                                ui.Rectangle(style={"background_color": 0xFF2A66FF})
                            ui.Label("LeftSimTarget live", word_wrap=True)
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
        messages = "paused" if not sim_playing else ("receiving" if packet_age is not None and packet_age <= float(controller.packet_stale_timeout_s) else "disconnected")
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
        joint_targets_deg = result.get("joint_targets_deg") or result.get("model_joint_targets_deg") or []
        joint_lower_limits_deg = result.get("joint_lower_limits_deg") or debug.get("joint_lower_limits_deg") or []
        joint_upper_limits_deg = result.get("joint_upper_limits_deg") or debug.get("joint_upper_limits_deg") or []
        stage_error_score = debug.get("stage_error_score") or (debug.get("result") or {}).get("stage_error_score")

        joint_rows = []
        if stage_joint_positions_deg:
            for idx, angle in enumerate(stage_joint_positions_deg):
                name = joint_names[idx] if idx < len(joint_names) else f"joint_{idx}"
                start_angle = stage_start_joint_positions_deg[idx] if stage_start_joint_positions_deg and idx < len(stage_start_joint_positions_deg) else angle
                lower_limit = float(joint_lower_limits_deg[idx]) if idx < len(joint_lower_limits_deg) else None
                upper_limit = float(joint_upper_limits_deg[idx]) if idx < len(joint_upper_limits_deg) else None
                weight = joint_weights[idx] if idx < len(joint_weights) else 0.0
                target_deg = float(joint_targets_deg[idx]) if idx < len(joint_targets_deg) else None
                joint_rows.append({
                    "name": name,
                    "start_text": f"{float(start_angle):+.1f}",
                    "min_text": "-" if lower_limit is None else f"{lower_limit:+.1f}",
                    "cur_text": f"{float(angle):+.1f}",
                    "tgt_text": "-" if target_deg is None else f"{target_deg:+.1f}",
                    "max_text": "-" if upper_limit is None else f"{upper_limit:+.1f}",
                    "wt_text": f"{float(weight):.2f}",
                    "current_deg": float(angle),
                    "target_deg": target_deg,
                    "lower_limit_deg": lower_limit,
                    "upper_limit_deg": upper_limit,
                })

        error_text = "-" if stage_error_score is None else f"mean={float(stage_error_score.get('mean_error_norm_m', 0.0)):.4f} latest={float(stage_error_score.get('latest_error_norm_m', 0.0)):.4f} n={int(stage_error_score.get('window_size', 0))}"
        packet_text = "-" if packet_age is None else f"{packet if packet is not None else '-'} ({packet_age:.2f}s ago)"

        self._labels["status"].text = status_line
        self._labels["messages"].text = messages
        self._labels["anchor"].text = anchor
        self._labels["grip"].text = f"{grip:.2f}"
        self._labels["trigger"].text = f"{trigger:.2f}"
        self._labels["buttons"].text = buttons
        self._labels["profile"].text = "hope_jr_sim_config.json"
        self._labels["error_score"].text = error_text
        self._labels["packet"].text = packet_text
        self._labels["mapped_delta"].text = mapped_text

        for idx in range(7):
            row = joint_rows[idx] if idx < len(joint_rows) else {
                "name": "-", "start_text": "-", "min_text": "-", "cur_text": "-", "tgt_text": "-", "max_text": "-", "wt_text": "-",
                "current_deg": None, "target_deg": None, "lower_limit_deg": None, "upper_limit_deg": None,
            }
            self._labels[f"joint_name_{idx}"].text = row["name"]
            self._labels[f"joint_start_{idx}"].text = row["start_text"]
            self._labels[f"joint_min_{idx}"].text = row["min_text"]
            self._labels[f"joint_cur_{idx}"].text = row["cur_text"]
            self._labels[f"joint_tgt_{idx}"].text = row["tgt_text"]
            self._labels[f"joint_max_{idx}"].text = row["max_text"]
            self._labels[f"joint_wt_{idx}"].text = row["wt_text"]
            joint_color = self._joint_current_color(
                current_deg=row["current_deg"],
                target_deg=row["target_deg"],
                lower_limit_deg=row["lower_limit_deg"],
                upper_limit_deg=row["upper_limit_deg"],
            )
            self._set_label_color(self._labels[f"joint_cur_{idx}"], joint_color)
            self._set_label_color(self._labels[f"joint_name_{idx}"], self._TEXT_NEUTRAL)
            self._set_label_color(self._labels[f"joint_start_{idx}"], self._TEXT_DIM)
            self._set_label_color(self._labels[f"joint_min_{idx}"], self._TEXT_DIM)
            self._set_label_color(self._labels[f"joint_tgt_{idx}"], self._TEXT_DIM)
            self._set_label_color(self._labels[f"joint_max_{idx}"], self._TEXT_DIM)
            self._set_label_color(self._labels[f"joint_wt_{idx}"], self._TEXT_DIM)
