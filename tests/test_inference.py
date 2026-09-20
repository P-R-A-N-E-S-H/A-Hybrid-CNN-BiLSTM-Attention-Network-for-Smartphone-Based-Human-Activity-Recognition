"""
======================================================================
test_inference.py - Unit Tests for Real-Time Inference Engine
======================================================================
"""

import sys
import os
import numpy as np
import pytest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from realtime_inference_engine import RealtimeInferenceEngine
import config


def test_realtime_inference_engine_prediction():
    model_path = os.path.join(config.TRAINED_MODELS_DIR, "proposed.keras")
    if not os.path.exists(model_path):
        pytest.skip("Trained model not found")

    engine = RealtimeInferenceEngine(model_name="proposed", backend="keras")
    dummy_window = np.random.normal(0, 1, (128, 9)).astype(np.float32)

    res = engine.predict_window(dummy_window)

    assert "activity" in res
    assert res["activity"] in config.ACTIVITY_NAMES
    assert "confidence" in res
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["probabilities"]) == 6
    assert "latency_ms" in res
    assert res["latency_ms"] > 0
