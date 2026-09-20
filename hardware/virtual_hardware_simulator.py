"""
======================================================================
virtual_hardware_simulator.py - Real-Time IMU Hardware Simulator
======================================================================

Generates continuous 50Hz IMU sensor telemetry (3-axis Accel + 3-axis Gyro)
simulating an ESP32/MPU-6050 physical wearable. Supports realistic kinematic
patterns for all 6 human activities (Walking, Walking Upstairs,
Walking Downstairs, Sitting, Standing, Laying) with Gaussian noise, gravity
vectors, and realistic step cadence.

Can also stream real test dataset sequences in a real-time playback loop.

Author: Pranesh
======================================================================
"""

import time
import threading
import math
import random
import numpy as np
from typing import Callable, Optional, Dict, Any, List
import os

from hardware.dsp_filter import HARSignalProcessor
import config


class VirtualHardwareSimulator:
    """
    High-fidelity physical IMU simulator for real-time demonstration
    and automated hardware pipeline testing.
    """

    def __init__(
        self,
        sampling_rate_hz: float = config.SAMPLING_RATE_HZ,
        on_window_ready: Optional[Callable[[np.ndarray, Dict[str, Any]], None]] = None,
        on_sample_emitted: Optional[Callable[[Dict[str, Any]], None]] = None,
        scaler_path: Optional[str] = None
    ):
        self.fs = sampling_rate_hz
        self.dt = 1.0 / self.fs
        self.on_window_ready = on_window_ready
        self.on_sample_emitted = on_sample_emitted

        if scaler_path is None:
            default_scaler = os.path.join(config.PROCESSED_DIR, "scaler.pkl")
            scaler_path = default_scaler if os.path.exists(default_scaler) else None

        self.dsp = HARSignalProcessor(
            sampling_rate_hz=self.fs,
            window_size=config.WINDOW_SIZE,
            window_step=config.WINDOW_STEP,
            scaler_path=scaler_path
        )

        self.is_running = False
        self.stream_thread: Optional[threading.Thread] = None
        self.current_activity = "WALKING"
        self.packet_count = 0
        self.start_time = 0.0
        self.last_reading: List[float] = [0.0] * 6

        # Internal kinematic state
        self._t = 0.0

    def set_activity(self, activity_name: str):
        """Changes the simulated human activity pattern."""
        if activity_name in config.ACTIVITY_NAMES:
            self.current_activity = activity_name
            print(f"[VirtualHardware] Switched simulated activity to: {activity_name}")
        else:
            print(f"[VirtualHardware Warning] Unknown activity '{activity_name}'")

    def _generate_synthetic_sample(self) -> np.ndarray:
        """
        Synthesizes a realistic 6-DOF IMU reading based on current activity kinematics.
        Units: Accel in g (~1.0g gravity vector), Gyro in rad/sec.
        """
        self._t += self.dt
        t = self._t
        noise = lambda scale=0.03: np.random.normal(0, scale)

        if self.current_activity == "WALKING":
            # Walking cadence ~ 1.8 Hz
            f_step = 1.8
            ax = 0.15 * math.sin(2 * math.pi * f_step * t) + noise(0.04)
            ay = 0.98 + 0.25 * math.sin(4 * math.pi * f_step * t) + noise(0.05)  # Vertical bounce
            az = 0.30 * math.cos(2 * math.pi * f_step * t) + noise(0.04)         # Forward/backward
            gx = 0.40 * math.sin(2 * math.pi * f_step * t) + noise(0.05)
            gy = 0.10 * math.cos(2 * math.pi * f_step * t) + noise(0.03)
            gz = 0.25 * math.sin(2 * math.pi * f_step * t + 0.5) + noise(0.04)

        elif self.current_activity == "WALKING_UPSTAIRS":
            # Steeper vertical impulse, slightly slower cadence ~ 1.4 Hz
            f_step = 1.4
            ax = 0.20 * math.sin(2 * math.pi * f_step * t) + noise(0.05)
            ay = 1.05 + 0.45 * math.sin(2 * math.pi * f_step * t) + noise(0.06)
            az = 0.40 * math.cos(2 * math.pi * f_step * t) + noise(0.05)
            gx = 0.60 * math.sin(2 * math.pi * f_step * t) + noise(0.06)
            gy = 0.15 * math.cos(2 * math.pi * f_step * t) + noise(0.04)
            gz = 0.35 * math.sin(2 * math.pi * f_step * t) + noise(0.05)

        elif self.current_activity == "WALKING_DOWNSTAIRS":
            # Higher impact peaks, cadence ~ 1.6 Hz
            f_step = 1.6
            ax = 0.22 * math.sin(2 * math.pi * f_step * t) + noise(0.05)
            ay = 0.95 + 0.55 * math.sin(2 * math.pi * f_step * t) + noise(0.08)
            az = 0.35 * math.cos(2 * math.pi * f_step * t) + noise(0.05)
            gx = 0.70 * math.sin(2 * math.pi * f_step * t) + noise(0.07)
            gy = 0.20 * math.cos(2 * math.pi * f_step * t) + noise(0.05)
            gz = 0.30 * math.sin(2 * math.pi * f_step * t) + noise(0.05)

        elif self.current_activity == "SITTING":
            # Upright posture (Y-axis ~ 0.98g), minimal movement
            ax = 0.05 + noise(0.008)
            ay = 0.98 + noise(0.009)
            az = 0.10 + noise(0.008)
            gx = noise(0.01)
            gy = noise(0.01)
            gz = noise(0.01)

        elif self.current_activity == "STANDING":
            # Upright posture with subtle postural sway (~0.2 Hz)
            f_sway = 0.25
            ax = 0.02 * math.sin(2 * math.pi * f_sway * t) + noise(0.012)
            ay = 1.00 + 0.01 * math.cos(2 * math.pi * f_sway * t) + noise(0.012)
            az = 0.04 * math.sin(2 * math.pi * f_sway * t + 1.0) + noise(0.012)
            gx = 0.03 * math.sin(2 * math.pi * f_sway * t) + noise(0.015)
            gy = 0.02 * math.cos(2 * math.pi * f_sway * t) + noise(0.015)
            gz = noise(0.015)

        elif self.current_activity == "LAYING":
            # Horizontal posture (Z-axis gravity or X-axis gravity ~ 0.98g, Y-axis near 0)
            ax = 0.02 + noise(0.005)
            ay = 0.05 + noise(0.005)
            az = 0.98 + noise(0.005)
            gx = noise(0.006)
            gy = noise(0.006)
            gz = noise(0.006)
        else:
            ax, ay, az = 0.0, 1.0, 0.0
            gx, gy, gz = 0.0, 0.0, 0.0

        return np.array([ax, ay, az, gx, gy, gz], dtype=np.float32)

    def start(self):
        """Starts real-time simulated telemetry generation."""
        if self.is_running:
            return
        self.is_running = True
        self.start_time = time.time()
        self.packet_count = 0
        self.stream_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.stream_thread.start()
        print(f"[VirtualHardware] Started continuous 50Hz simulator. Initial activity: {self.current_activity}")

    def _run_loop(self):
        """Precise 50Hz loop."""
        interval = 1.0 / self.fs
        next_time = time.perf_counter()

        while self.is_running:
            raw_reading = self._generate_synthetic_sample()
            self.packet_count += 1
            self.last_reading = raw_reading.tolist()

            # Emit single sample telemetry if registered
            if self.on_sample_emitted:
                self.on_sample_emitted({
                    "timestamp": time.time(),
                    "activity": self.current_activity,
                    "ax": float(raw_reading[0]),
                    "ay": float(raw_reading[1]),
                    "az": float(raw_reading[2]),
                    "gx": float(raw_reading[3]),
                    "gy": float(raw_reading[4]),
                    "gz": float(raw_reading[5]),
                    "packet_id": self.packet_count
                })

            # Feed to DSP pipeline
            ready_windows = self.dsp.add_streaming_samples(raw_reading)
            for window in ready_windows:
                if self.on_window_ready:
                    meta = {
                        "source": "virtual_simulator",
                        "simulated_ground_truth": self.current_activity,
                        "timestamp": time.time(),
                        "packet_count": self.packet_count
                    }
                    self.on_window_ready(window, meta)

            next_time += interval
            sleep_duration = next_time - time.perf_counter()
            if sleep_duration > 0:
                time.sleep(sleep_duration)
            else:
                next_time = time.perf_counter()

    def get_stats(self) -> Dict[str, Any]:
        """Returns simulator telemetry statistics."""
        elapsed = max(time.time() - self.start_time, 0.001)
        fps = self.packet_count / elapsed if self.is_running else 0.0
        return {
            "is_simulating": self.is_running,
            "current_activity": self.current_activity,
            "packets_generated": self.packet_count,
            "effective_sampling_rate_hz": round(fps, 2),
            "last_reading": self.last_reading,
            "uptime_sec": round(elapsed, 1)
        }

    def stop(self):
        """Stops the simulator."""
        self.is_running = False
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=1.0)
        print("[VirtualHardware] Simulator stopped.")


if __name__ == "__main__":
    def on_win(win, meta):
        print(f"[Sim Window] Extracted shape: {win.shape}, Ground Truth: {meta['simulated_ground_truth']}")

    sim = VirtualHardwareSimulator(on_window_ready=on_win)
    sim.start()
    time.sleep(3)
    sim.set_activity("WALKING_UPSTAIRS")
    time.sleep(3)
    sim.stop()
