"""
======================================================================
test_quantization.py - Unit Tests for Edge AI Quantization
======================================================================
"""

import sys
import os
import numpy as np
import pytest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import config
from edge_deployment.tflite_to_c_header import tflite_to_c_array


def test_tflite_to_c_array_generation(tmp_path):
    # Create dummy binary file
    dummy_bin = tmp_path / "test_model.tflite"
    dummy_bin.write_bytes(b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09")

    out_h = tmp_path / "test_model.h"
    res_path = tflite_to_c_array(str(dummy_bin), str(out_h), array_name="test_model_data")

    assert os.path.exists(res_path)
    content = out_h.read_text()
    assert "test_model_data" in content
    assert "test_model_data_len" in content
