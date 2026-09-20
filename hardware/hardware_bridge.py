"""
======================================================================
hardware_bridge.py - Real-Time Hardware Communication & Sensor Ingestion
======================================================================

Connects to physical microcontrollers (ESP32 / Arduino + MPU6050) over USB-Serial,
parses high-rate 50Hz IMU telemetry packets, applies digital signal processing,
and feeds a continuous sliding window buffer for real-time model inference.

Author: Pranesh
======================================================================
"""

import sys
import os
import time
import threading
import serial
import serial.tools.list_ports
import numpy as np
from typing import Callable, Optional, List, Dict, Any

from hardware.dsp_filter import HARSignalProcessor
import config


class SerialHardwareBridge:
    """
    Asynchronous USB-Serial bridge for real-time IMU hardware streaming.
    """

    def __init__(
        self,
        port: Optional[str] = None,
        baudrate: int = config.SERIAL_BAUD_RATE,
        on_window_ready: Optional[Callable[[np.ndarray, Dict[str, Any]], None]] = None,
        scaler_path: Optional[str] = None
    ):
        self.port = port
        self.baudrate = baudrate
        self.on_window_ready = on_window_ready

        # Path to pre-trained scaler
        if scaler_path is None:
            default_scaler = os.path.join(config.PROCESSED_DIR, "scaler.pkl")
            scaler_path = default_scaler if os.path.exists(default_scaler) else None

        self.dsp = HARSignalProcessor(
            sampling_rate_hz=config.SAMPLING_RATE_HZ,
            window_size=config.WINDOW_SIZE,
            window_step=config.WINDOW_STEP,
            scaler_path=scaler_path
        )

        self.serial_conn: Optional[serial.Serial] = None
        self.is_running = False
        self.reader_thread: Optional[threading.Thread] = None

        # Statistics
        self.packet_count = 0
        self.dropped_count = 0
        self.start_time = 0.0
        self.last_packet_time = 0.0
        self.last_raw_reading: List[float] = [0.0] * 6

    @staticmethod
    def list_available_ports() -> List[Dict[str, str]]:
        """Scans and returns available COM/Serial ports."""
        ports = serial.tools.list_ports.comports()
        port_list = []
        for p in ports:
            port_list.append({
                "device": p.device,
                "description": p.description,
                "hwid": p.hwid
            })
        return port_list

    def auto_detect_port(self) -> Optional[str]:
        """Attempts to auto-detect an ESP32 or Arduino device."""
        ports = serial.tools.list_ports.comports()
        for p in ports:
            desc = p.description.lower()
            if any(keyword in desc for keyword in ["ch340", "cp210", "ftdi", "usb serial", "arduino", "esp32"]):
                return p.device
        if len(ports) > 0:
            return ports[0].device
        return None

    def connect(self) -> bool:
        """Opens the serial connection."""
        if not self.port:
            self.port = self.auto_detect_port()
            if not self.port:
                print("[HardwareBridge Error] No active COM port found.")
                return False

        try:
            print(f"[HardwareBridge] Connecting to {self.port} @ {self.baudrate} baud...")
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1.0,
                rtscts=False,
                dsrdtr=False
            )
            # Flush existing buffer
            time.sleep(1.5)  # Allow microcontroller reboot/settle
            self.serial_conn.reset_input_buffer()
            self.start_time = time.time()
            print(f"[HardwareBridge] Successfully connected to {self.port}!")
            return True
        except Exception as e:
            print(f"[HardwareBridge Connection Error] {e}")
            self.serial_conn = None
            return False

    def start_streaming(self):
        """Starts background reader thread."""
        if self.is_running:
            return
        if self.serial_conn is None or not self.serial_conn.is_open:
            if not self.connect():
                return

        self.is_running = True
        self.reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.reader_thread.start()
        print("[HardwareBridge] Background streaming thread active.")

    def _read_loop(self):
        """Continuous packet ingestion loop."""
        while self.is_running and self.serial_conn and self.serial_conn.is_open:
            try:
                line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                if not line or line.startswith("#"):
                    continue  # Comment or header line

                parts = line.split(',')
                if len(parts) >= 6:
                    ax = float(parts[0])
                    ay = float(parts[1])
                    az = float(parts[2])
                    gx = float(parts[3])
                    gy = float(parts[4])
                    gz = float(parts[5])

                    self.packet_count += 1
                    self.last_packet_time = time.time()
                    self.last_raw_reading = [ax, ay, az, gx, gy, gz]

                    sample = np.array([[ax, ay, az, gx, gy, gz]], dtype=np.float32)
                    ready_windows = self.dsp.add_streaming_samples(sample)

                    for window in ready_windows:
                        if self.on_window_ready:
                            meta = {
                                "source": "serial",
                                "port": self.port,
                                "timestamp": time.time(),
                                "packet_count": self.packet_count
                            }
                            self.on_window_ready(window, meta)
                else:
                    self.dropped_count += 1
            except Exception as e:
                self.dropped_count += 1
                continue

    def get_stats(self) -> Dict[str, Any]:
        """Returns streaming telemetry statistics."""
        now = time.time()
        elapsed = max(now - self.start_time, 0.001)
        fps = self.packet_count / elapsed if self.is_running else 0.0
        return {
            "port": self.port,
            "connected": self.is_running and (self.serial_conn is not None and self.serial_conn.is_open),
            "packets_received": self.packet_count,
            "packets_dropped": self.dropped_count,
            "effective_sampling_rate_hz": round(fps, 2),
            "last_reading": self.last_raw_reading,
            "uptime_sec": round(elapsed, 1)
        }

    def stop_streaming(self):
        """Terminates streaming thread and closes port."""
        self.is_running = False
        if self.reader_thread and self.reader_thread.is_alive():
            self.reader_thread.join(timeout=1.0)
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("[HardwareBridge] Serial connection closed.")
        self.serial_conn = None


if __name__ == "__main__":
    print("=== Hardware Bridge Diagnostics ===")
    ports = SerialHardwareBridge.list_available_ports()
    print("Available Ports:", ports)

    def sample_callback(window: np.ndarray, meta: Dict[str, Any]):
        print(f"[Callback] Window ready with shape {window.shape} at {meta['timestamp']}")

    bridge = SerialHardwareBridge(on_window_ready=sample_callback)
    if bridge.connect():
        bridge.start_streaming()
        try:
            for _ in range(5):
                time.sleep(1)
                print("Stats:", bridge.get_stats())
        finally:
            bridge.stop_streaming()
    else:
        print("No physical hardware detected. Use VirtualHardwareSimulator for live demonstrations.")
