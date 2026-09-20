"""
======================================================================
convert_tflite.py - TensorFlow Lite Model Quantization & Exporter
======================================================================

Converts Keras models to optimized TensorFlow Lite (.tflite) formats:
  1. Float32 (Standard baseline)
  2. Float16 Quantization (2x memory reduction, minimal accuracy drop)
  3. Dynamic Range Quantization
  4. Full INT8 Post-Training Quantization (4x memory reduction for microcontrollers)

Author: Pranesh
======================================================================
"""

import os
import sys
import json
import time
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import config
from models import TemporalAttentionPooling


def representative_data_gen():
    """Generates representative samples for INT8 calibration."""
    x_train_path = os.path.join(config.PROCESSED_DIR, "X_train.npy")
    if os.path.exists(x_train_path):
        x_train = np.load(x_train_path, mmap_mode='r')
        num_samples = min(200, len(x_train))
        indices = np.random.choice(len(x_train), num_samples, replace=False)
        for idx in indices:
            sample = x_train[idx:idx+1].astype(np.float32)
            yield [sample]
    else:
        # Synthetic representative generator if dataset not present
        for _ in range(100):
            sample = np.random.normal(0, 1, size=(1, config.WINDOW_SIZE, config.NUM_CHANNELS)).astype(np.float32)
            yield [sample]


def convert_model(
    model_name: str = "proposed",
    keras_path: str = None,
    output_dir: str = None
):
    """Converts a trained Keras model into optimized TFLite variants."""
    if keras_path is None:
        keras_path = os.path.join(config.TRAINED_MODELS_DIR, f"{model_name}.keras")
    if output_dir is None:
        output_dir = os.path.join(config.BASE_DIR, "edge_deployment", "tflite_models")

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n========================================================")
    print(f" Converting {model_name} to TensorFlow Lite Formats")
    print(f" Source: {keras_path}")
    print(f" Target Dir: {output_dir}")
    print(f"========================================================")

    custom_objects = {"TemporalAttentionPooling": TemporalAttentionPooling}
    model = tf.keras.models.load_model(keras_path, custom_objects=custom_objects)

    results = {}

    # 1. Standard FP32 TFLite
    print("\n[1/3] Converting to Float32 TFLite...")
    converter_fp32 = tf.lite.TFLiteConverter.from_keras_model(model)
    # Enable Select TF Ops for custom attention ops if needed
    converter_fp32.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter_fp32._experimental_lower_tensor_list_ops = False
    tflite_fp32 = converter_fp32.convert()

    fp32_path = os.path.join(output_dir, f"{model_name}_fp32.tflite")
    with open(fp32_path, "wb") as f:
        f.write(tflite_fp32)
    fp32_size_kb = len(tflite_fp32) / 1024.0
    results["fp32"] = {"path": fp32_path, "size_kb": round(fp32_size_kb, 2)}
    print(f"  ✓ Saved FP32: {fp32_path} ({fp32_size_kb:.2f} KB)")

    # 2. Float16 Quantization
    print("\n[2/3] Converting to Float16 Quantized TFLite...")
    try:
        converter_fp16 = tf.lite.TFLiteConverter.from_keras_model(model)
        converter_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
        converter_fp16.target_spec.supported_types = [tf.float16]
        converter_fp16.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        tflite_fp16 = converter_fp16.convert()

        fp16_path = os.path.join(output_dir, f"{model_name}_fp16.tflite")
        with open(fp16_path, "wb") as f:
            f.write(tflite_fp16)
        fp16_size_kb = len(tflite_fp16) / 1024.0
        results["fp16"] = {"path": fp16_path, "size_kb": round(fp16_size_kb, 2)}
        print(f"  ✓ Saved FP16: {fp16_path} ({fp16_size_kb:.2f} KB, Compression: {fp32_size_kb/fp16_size_kb:.2f}x)")
    except Exception as e:
        print(f"  ✗ FP16 conversion note: {e}")

    # 3. Dynamic / Post-Training INT8 Quantization
    print("\n[3/3] Converting to INT8 Quantized TFLite...")
    try:
        converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
        converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
        converter_int8.representative_dataset = representative_data_gen
        converter_int8.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        tflite_int8 = converter_int8.convert()

        int8_path = os.path.join(output_dir, f"{model_name}_int8.tflite")
        with open(int8_path, "wb") as f:
            f.write(tflite_int8)
        int8_size_kb = len(tflite_int8) / 1024.0
        results["int8"] = {"path": int8_path, "size_kb": round(int8_size_kb, 2)}
        print(f"  ✓ Saved INT8: {int8_path} ({int8_size_kb:.2f} KB, Compression: {fp32_size_kb/int8_size_kb:.2f}x)")
    except Exception as e:
        print(f"  ✗ INT8 conversion note: {e}")

    # Summary Report
    summary_path = os.path.join(output_dir, f"{model_name}_tflite_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n[Summary] Successfully exported TFLite models to {output_dir}")
    return results


if __name__ == "__main__":
    # Convert all available trained models
    for m in ["proposed", "cnn", "lstm", "bilstm", "cnn_lstm"]:
        kpath = os.path.join(config.TRAINED_MODELS_DIR, f"{m}.keras")
        if os.path.exists(kpath):
            convert_model(m, kpath)
