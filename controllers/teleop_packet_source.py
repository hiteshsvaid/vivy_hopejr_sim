#!/usr/bin/env python3

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any


class TeleopPacketSource:
    def __init__(self, *, packet_path: Path, use_udp: bool, udp_listen_host: str, udp_listen_port: int):
        self.packet_path = Path(packet_path)
        self.use_udp = bool(use_udp)
        self.udp_listen_host = udp_listen_host
        self.udp_listen_port = int(udp_listen_port)
        self._udp_socket = None
        if self.use_udp:
            self._udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._udp_socket.bind((self.udp_listen_host, self.udp_listen_port))
            self._udp_socket.setblocking(False)

    def read_latest_packet(self) -> dict[str, Any] | None:
        if self._udp_socket is not None:
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
        if not self.packet_path.is_file():
            return None
        return json.loads(self.packet_path.read_text())
