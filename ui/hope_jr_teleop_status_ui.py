#!/usr/bin/env python3

import time
from typing import Any


class HopeJrTeleopStatusUi:
    def __init__(self, *, width: int = 720, height: int = 400):
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
            with ui.ScrollingFrame(horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                                   vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED):
                with ui.VStack(spacing=3):
                    label_style = {"color": 0xFFB8B8B8, "font_size": 13}
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

                    with ui.VStack(spacing=1):
                        with ui.VStack(spacing=0):
                            ui.Label("ADVISORY SOURCE", style=section_style)
                            self._labels["advisory_source"] = ui.Label("-", word_wrap=True)
                        with ui.VStack(spacing=0):
                            ui.Label("ADVISORY SEV", style=section_style)
                            self._labels["advisory_severity"] = ui.Label("-", word_wrap=True)
                        with ui.VStack(spacing=0):
                            ui.Label("ADVISORY JOINT", style=section_style)
                            self._labels["advisory_joint"] = ui.Label("-", word_wrap=True)
                        with ui.VStack(spacing=0):
                            ui.Label("ADVISORY ANGLES", style=section_style)
                            self._labels["advisory_angles"] = ui.Label("-", word_wrap=True)
                        with ui.VStack(spacing=0):
                            ui.Label("ADVISORY LIMITS", style=section_style)
                            self._labels["advisory_limits"] = ui.Label("-", word_wrap=True)
                        with ui.VStack(spacing=0):
                            ui.Label("ADVISORY WHY", style=section_style)
                            self._labels["advisory_reasons"] = ui.Label("-", word_wrap=True)
                        with ui.VStack(spacing=0):
                            ui.Label("ADVISORY DO", style=section_style)
                            self._labels["advisory_recommendations"] = ui.Label("-", word_wrap=True)

                    ui.Line(style={"color": 0x55FFFFFF})

                    with ui.VStack(spacing=2):
                        with ui.VGrid(column_count=5, row_height=18, column_widths=[56, 56, 56, 56, 42], spacing=6):
                            self._labels["joint_header_start"] = ui.Label("start", style=label_style)
                            self._labels["joint_header_min"] = ui.Label("min", style=label_style)
                            self._labels["joint_header_cur"] = ui.Label("cur", style=label_style)
                            self._labels["joint_header_max"] = ui.Label("max", style=label_style)
                            self._labels["joint_header_wt"] = ui.Label("wt", style=label_style)
                        for idx in range(7):
                            with ui.VStack(spacing=0):
                                self._labels[f"joint_name_{idx}"] = ui.Label("-", word_wrap=True)
                                with ui.VGrid(column_count=5, row_height=18, column_widths=[56, 56, 56, 56, 42], spacing=6):
                                    self._labels[f"joint_start_{idx}"] = ui.Label("-")
                                    self._labels[f"joint_min_{idx}"] = ui.Label("-")
                                    self._labels[f"joint_cur_{idx}"] = ui.Label("-")
                                    self._labels[f"joint_max_{idx}"] = ui.Label("-")
                                    self._labels[f"joint_wt_{idx}"] = ui.Label("-")

                    ui.Line(style={"color": 0x55FFFFFF})

                    with ui.VStack(spacing=2):
                        ui.Label("MARKERS", style=section_style)
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
        joint_lower_limits_deg = result.get("joint_lower_limits_deg") or debug.get("joint_lower_limits_deg") or []
        joint_upper_limits_deg = result.get("joint_upper_limits_deg") or debug.get("joint_upper_limits_deg") or []
        joint_rows = []
        if stage_joint_positions_deg:
            for idx, angle in enumerate(stage_joint_positions_deg):
                name = joint_names[idx] if idx < len(joint_names) else f"joint_{idx}"
                start_angle = (
                    stage_start_joint_positions_deg[idx]
                    if stage_start_joint_positions_deg and idx < len(stage_start_joint_positions_deg)
                    else angle
                )
                lower_limit = joint_lower_limits_deg[idx] if idx < len(joint_lower_limits_deg) else None
                upper_limit = joint_upper_limits_deg[idx] if idx < len(joint_upper_limits_deg) else None
                weight = joint_weights[idx] if idx < len(joint_weights) else 0.0
                joint_rows.append((
                    name,
                    f"{float(start_angle):+.1f}",
                    "-" if lower_limit is None else f"{float(lower_limit):+.1f}",
                    f"{float(angle):+.1f}",
                    "-" if upper_limit is None else f"{float(upper_limit):+.1f}",
                    f"{float(weight):.2f}",
                ))
        stage_profile = "hope_jr_sim_config.json"
        teleop_safety_advisory = debug.get("teleop_safety_advisory") or result.get("teleop_safety_advisory") or {}
        active_advisory = teleop_safety_advisory.get("active", teleop_safety_advisory) if isinstance(teleop_safety_advisory, dict) else {}
        advisory_snapshot = teleop_safety_advisory.get("joint_limit_snapshot", {}) if isinstance(teleop_safety_advisory, dict) else {}
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

        if not active_advisory:
            advisory_source = "-"
            advisory_severity = "-"
            advisory_joint = "-"
            advisory_angles = "-"
            advisory_limits = "-"
            advisory_reasons = "-"
            advisory_recommendations = "-"
        else:
            advisory_source = str(active_advisory.get("source_label") or active_advisory.get("source") or "-")
            advisory_severity = str(active_advisory.get("severity", "-"))
            advisory_joint = str(active_advisory.get("joint_name") or active_advisory.get("joint_step_abs_max_joint") or "-")
            current_joint = advisory_snapshot.get("current_joint_deg")
            target_joint = active_advisory.get("target_joint_deg", advisory_snapshot.get("target_joint_deg"))
            advisory_angles = (
                f"cur={float(current_joint):+.1f} tgt={float(target_joint):+.1f}"
                if current_joint is not None and target_joint is not None
                else "-"
            )
            lower_limit = active_advisory.get("lower_limit_deg", advisory_snapshot.get("lower_limit_deg"))
            upper_limit = active_advisory.get("upper_limit_deg", advisory_snapshot.get("upper_limit_deg"))
            lower_margin = active_advisory.get("lower_margin_deg")
            upper_margin = active_advisory.get("upper_margin_deg")
            advisory_limits = (
                f"[{float(lower_limit):+.1f}, {float(upper_limit):+.1f}] lm={float(lower_margin):+.1f} um={float(upper_margin):+.1f}"
                if None not in (lower_limit, upper_limit, lower_margin, upper_margin)
                else "-"
            )
            advisory_reasons = ", ".join(active_advisory.get("reasons", [])) or "-"
            advisory_recommendations = ", ".join(active_advisory.get("recommendations", [])[:2]) or "-"

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
        self._labels["advisory_source"].text = advisory_source
        self._labels["advisory_severity"].text = advisory_severity
        self._labels["advisory_joint"].text = advisory_joint
        self._labels["advisory_angles"].text = advisory_angles
        self._labels["advisory_limits"].text = advisory_limits
        self._labels["advisory_reasons"].text = advisory_reasons
        self._labels["advisory_recommendations"].text = advisory_recommendations
        for idx in range(7):
            if idx < len(joint_rows):
                name, start_text, min_text, cur_text, max_text, wt_text = joint_rows[idx]
            else:
                name, start_text, min_text, cur_text, max_text, wt_text = ("-", "-", "-", "-", "-", "-")
            self._labels[f"joint_name_{idx}"].text = name
            self._labels[f"joint_start_{idx}"].text = start_text
            self._labels[f"joint_min_{idx}"].text = min_text
            self._labels[f"joint_cur_{idx}"].text = cur_text
            self._labels[f"joint_max_{idx}"].text = max_text
            self._labels[f"joint_wt_{idx}"].text = wt_text
