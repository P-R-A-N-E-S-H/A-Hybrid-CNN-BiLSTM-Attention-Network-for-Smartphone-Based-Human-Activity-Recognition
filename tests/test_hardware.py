"""
======================================================================
test_hardware.py - Unit Tests for Virtual Simulator & Hardware Bridge
======================================================================
"""

import sys
import time
import numpy as np
import pytest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from hardware.virtual_hardware_simulator import VirtualHardwareSimulator
from hardware.hardware_bridge import SerialHardwareBridge
import config


def test_virtual_simulator_sample_generation():
    sim = VirtualHardwareSimulator()
    sample = sim._generate_synthetic_sample()

    assert sample.shape == (6,)
    # Ax, Ay, Az, Gx, Gy, Gz must all be finite numbers
    assert np.all(np.isfinite(sample))


def test_virtual_simulator_activity_switching():
    sim = VirtualHardwareSimulator()
    sim.set_activity("WALKING_UPSTAIRS")
    assert sim.current_activity == "WALKING_UPSTAIRS"

    sim.set_activity("LAYING")
    assert sim.current_activity == "LAYING"


def test_virtual_simulator_streaming_callback():
    emitted_samples = []
    ready_windows = []

    def on_sample(s):
        emitted_samples.append(s)

    def on_window(w, meta):
        ready_windows.append((w, meta))

    sim = VirtualHardwareSimulator(
        sampling_rate_hz=50.0,
        on_window_ready=on_window,
        on_sample_emitted=on_sample
    )

    sim.start()
    time.sleep(0.5)  # Run for 0.5s (~25 samples)
    sim.stop()

    assert len(emitted_samples) > 10
    assert "ax" in emitted_samples[0]
    assert "activity" in emitted_samples[0]


def test_serial_bridge_port_listing():
    ports = SerialHardwareBridge.list_available_ports()
    assert isinstance(ports, list)
