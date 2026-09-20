"""
======================================================================
test_dsp.py - Unit Tests for Digital Signal Processing Pipeline
======================================================================
"""

import sys
import numpy as np
import pytest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from hardware.dsp_filter import HARSignalProcessor
import config


def test_dsp_initialization():
    processor = HARSignalProcessor(
        sampling_rate_hz=50.0,
        window_size=128,
        window_step=64
    )
    assert processor.fs == 50.0
    assert processor.window_size == 128
    assert processor.window_step == 64


def test_gravity_separation():
    processor = HARSignalProcessor()
    # Simulated 128-sample total acceleration with 1.0g gravity + 1.5Hz body oscillation
    t = np.linspace(0, 2.56, 128)
    gravity_true = np.ones((128, 3)) * 0.98
    body_true = np.sin(2 * np.pi * 1.5 * t)[:, None] * np.array([0.2, 0.4, 0.1])
    total_accel = gravity_true + body_true

    body_est, gravity_est = processor.separate_gravity_and_body(total_accel)

    assert body_est.shape == (128, 3)
    assert gravity_est.shape == (128, 3)
    # Check that estimated gravity mean is close to 0.98g
    assert np.allclose(np.mean(gravity_est, axis=0), 0.98, atol=0.1)


def test_process_raw_6axis_window():
    processor = HARSignalProcessor()
    raw_window = np.random.normal(0, 1, (128, 6)).astype(np.float32)

    nine_ch = processor.process_raw_6axis_window(raw_window)
    assert nine_ch.shape == (128, 9)
    assert not np.isnan(nine_ch).any()


def test_sliding_window_buffer():
    processor = HARSignalProcessor(window_size=128, window_step=64)
    # Feed 100 samples -> No window ready yet (< 128)
    s1 = np.random.normal(0, 1, (100, 6)).astype(np.float32)
    w1 = processor.add_streaming_samples(s1)
    assert len(w1) == 0

    # Feed 50 more samples (Total 150) -> 1 window ready (128 samples), remaining 150-64 = 86
    s2 = np.random.normal(0, 1, (50, 6)).astype(np.float32)
    w2 = processor.add_streaming_samples(s2)
    assert len(w2) == 1
    assert w2[0].shape == (128, 9)
