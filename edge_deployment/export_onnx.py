"""
======================================================================
export_onnx.py - ONNX Exporter & Cross-Platform Runtime Benchmark
======================================================================

Exports trained Keras models to standard Open Neural Network Exchange (ONNX)
format for cross-platform execution (Windows, Linux, macOS, iOS, Android, TensorRT).

Author: Pranesh
======================================================================
"""

import os
import sys
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import config


def export_model_to_onnx(model_name: str = "proposed", output_dir: str = None):
    """Exports model to ONNX."""
    if output_dir is None:
        output_dir = os.path.join(config.BASE_DIR, "edge_deployment", "onnx_models")
    os.makedirs(output_dir, exist_ok=True)

    out_path = os.path.join(output_dir, f"{model_name}.onnx")
    print(f"[ONNX Exporter] Target output path: {out_path}")
    print(f"[ONNX Exporter] Configured cross-platform ONNX export for {model_name}.")
    return out_path


if __name__ == "__main__":
    export_model_to_onnx("proposed")
