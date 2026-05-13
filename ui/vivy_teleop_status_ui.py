#!/usr/bin/env python3

from ui.hope_jr_teleop_bottom_panel import HopeJrControlProfilePanel
from ui.hope_jr_teleop_side_panel import HopeJrTeleopSidePanel


class HopeJrTeleopStatusUi:
    def __init__(self, *, width: int = 720, height: int = 400):
        self.side_panel = HopeJrTeleopSidePanel(width=width, height=height)
        self.profile_panel = HopeJrControlProfilePanel()

    def update(self, controller, debug=None) -> None:
        self.side_panel.update(controller, debug)
        self.profile_panel.update(controller, debug)
