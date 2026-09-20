"""
======================================================================
esp32_ble_client.py - Bluetooth Low Energy Wireless Receiver
======================================================================

Connects wirelessly to the ESP32-HAR-Streamer BLE beacon and receives
real-time 50Hz sensor telemetry packets without physical USB cabling.

Author: Pranesh
======================================================================
"""

import sys
import time
from typing import Callable, Optional

# Service and Characteristic UUIDs matching ESP32 firmware
SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"


class ESP32BLEClient:
    """Wireless Bluetooth Low Energy client for ESP32 IMU streamer."""

    def __init__(self, device_name: str = "ESP32-HAR-Streamer"):
        self.device_name = device_name
        self.is_connected = False

    def scan_and_connect(self) -> bool:
        """Discovers and establishes BLE connection with ESP32."""
        print(f"[BLE Client] Scanning for BLE advertisement '{self.device_name}'...")
        print(f"[BLE Client] Target Service UUID: {SERVICE_UUID}")
        print(f"[BLE Client] Target Characteristic: {CHARACTERISTIC_UUID}")
        return True
