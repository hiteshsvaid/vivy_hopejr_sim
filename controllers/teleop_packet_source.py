#!/usr/bin/env python3

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any


class TeleopPacketSource:
    def __init__(self, *, use_udp: bool, udp_listen_host: str, udp_listen_port: int):
        if not bool(use_udp):
            raise ValueError("TeleopPacketSource requires use_udp=True; file fallback was removed")
        self.udp_listen_host = udp_listen_host
        self.udp_listen_port = int(udp_listen_port)
        self._udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp_socket.bind((self.udp_listen_host, self.udp_listen_port))
        self._udp_socket.setblocking(False)

    def read_latest_packet(self) -> dict[str, Any] | None:
        latest_payload = None
        while True:
            try:
                payload, _addr = self._udp_socket.recvfrom(1024 * 1024)
            except BlockingIOError:
                break
            latest_payload = payload
        if latest_payload is None:
            return None
        return json.loads(latest_payload.decode("utf-8"))
