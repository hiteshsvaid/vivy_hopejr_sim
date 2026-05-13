#!/usr/bin/env python3

from ui.vivy_teleop_bottom_panel import VivyControlProfilePanel
from ui.vivy_teleop_side_panel import VivyTeleopSidePanel


class VivyTeleopStatusUi:
    def __init__(self, *, width: int = 720, height: int = 400):
        self.side_panel = VivyTeleopSidePanel(width=width, height=height)
        self.profile_panel = VivyControlProfilePanel()

    def update(self, controller, debug=None) -> None:
        self.side_panel.update(controller, debug)
        self.profile_panel.update(controller, debug)
