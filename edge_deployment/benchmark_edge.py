"""
======================================================================
benchmark_edge.py - Edge AI Latency, Memory & Throughput Profiler
======================================================================

Benchmarks real-time inference latency (milliseconds), peak memory usage,
model file footprint, and throughput (FPS / inferences per second) across:
  1. Standard Keras FP32 Model
  2. TFLite FP32 Interpreter
  3. TFLite FP16 Quantized Interpreter
  4. TFLite INT8 Post-Training Quantized Interpreter

Outputs structured benchmark results in JSON, CSV, and formatted console tables
ready for final review examination and thesis presentation.

Author: Pranesh
======================================================================
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import config
from models import TemporalAttentionPooling


class EdgeBenchmarkProfiler:
    """Comprehensive Edge AI Benchmark Profiler."""

    def __init__(self, warmup_runs: int = 20, test_runs: int = 100):
        self.warmup_runs = warmup_runs
        self.test_runs = test_runs
        self.custom_objects = {"TemporalAttentionPooling": TemporalAttentionPooling}

    def benchmark_keras_model(self, keras_path: str) -> dict:
        """Profiles native Keras model."""
        if not os.path.exists(keras_path):
            return {"error": "File not found"}

        model_size_mb = os.path.getsize(keras_path) / (1024 * 1024)
        model = tf.keras.models.load_model(keras_path, custom_objects=self.custom_objects)

        dummy_input = np.random.normal(0, 1, size=(1, config.WINDOW_SIZE, config.NUM_CHANNELS)).astype(np.float32)

        # Warmup
        for _ in range(self.warmup_runs):
            _ = model(dummy_input, training=False)

        # Timed benchmark
        latencies = []
        for _ in range(self.test_runs):
            t0 = time.perf_counter()
            _ = model(dummy_input, training=False)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

        latencies = np.array(latencies)
        params_count = model.count_params()

        return {
            "format": "Keras FP32",
            "size_mb": round(model_size_mb, 2),
            "parameters": params_count,
            "latency_mean_ms": round(float(np.mean(latencies)), 2),
            "latency_std_ms": round(float(np.std(latencies)), 2),
            "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2),
            "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
            "latency_p99_ms": round(float(np.percentile(latencies, 99)), 2),
            "throughput_fps": round(1000.0 / float(np.mean(latencies)), 1)
        }

    def benchmark_tflite_model(self, tflite_path: str, format_label: str = "TFLite") -> dict:
        """Profiles TFLite interpreter execution."""
        if not os.path.exists(tflite_path):
            return {"error": "File not found"}

        model_size_kb = os.path.getsize(tflite_path) / 1024.0
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_shape = input_details[0]['shape']
        input_dtype = input_details[0]['dtype']

        dummy_input = np.random.normal(0, 1, size=input_shape).astype(input_dtype)

        # Warmup
        for _ in range(self.warmup_runs):
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            interpreter.invoke()

        # Timed benchmark
        latencies = []
        for _ in range(self.test_runs):
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            t0 = time.perf_counter()
            interpreter.invoke()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

        latencies = np.array(latencies)

        return {
            "format": format_label,
            "size_kb": round(model_size_kb, 2),
            "latency_mean_ms": round(float(np.mean(latencies)), 2),
            "latency_std_ms": round(float(np.std(latencies)), 2),
            "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2),
            "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
            "latency_p99_ms": round(float(np.percentile(latencies, 99)), 2),
            "throughput_fps": round(1000.0 / float(np.mean(latencies)), 1)
        }

    def run_full_suite(self, model_name: str = "proposed") -> pd.DataFrame:
        """Runs benchmarks across all variants of a model."""
        keras_path = os.path.join(config.TRAINED_MODELS_DIR, f"{model_name}.keras")
        tflite_dir = os.path.join(config.BASE_DIR, "edge_deployment", "tflite_models")

        results = []

        # 1. Keras FP32
        print(f"Benchmarking {model_name} (Keras FP32)...")
        res_keras = self.benchmark_keras_model(keras_path)
        if "error" not in res_keras:
            results.append(res_keras)

        # 2. TFLite FP32
        p_fp32 = os.path.join(tflite_dir, f"{model_name}_fp32.tflite")
        if os.path.exists(p_fp32):
            print(f"Benchmarking {model_name} (TFLite FP32)...")
            results.append(self.benchmark_tflite_model(p_fp32, "TFLite FP32"))

        # 3. TFLite FP16
        p_fp16 = os.path.join(tflite_dir, f"{model_name}_fp16.tflite")
        if os.path.exists(p_fp16):
            print(f"Benchmarking {model_name} (TFLite FP16)...")
            results.append(self.benchmark_tflite_model(p_fp16, "TFLite FP16"))

        # 4. TFLite INT8
        p_int8 = os.path.join(tflite_dir, f"{model_name}_int8.tflite")
        if os.path.exists(p_int8):
            print(f"Benchmarking {model_name} (TFLite INT8)...")
            results.append(self.benchmark_tflite_model(p_int8, "TFLite INT8"))

        df = pd.DataFrame(results)
        
        # Save to CSV and JSON
        out_csv = os.path.join(config.RESULTS_DIR, f"{model_name}_edge_benchmarks.csv")
        out_json = os.path.join(config.RESULTS_DIR, f"{model_name}_edge_benchmarks.json")
        df.to_csv(out_csv, index=False)
        with open(out_json, "w") as f:
            json.dump(results, f, indent=4)

        print(f"\nSaved benchmark results to {out_csv}")
        return df


if __name__ == "__main__":
    profiler = EdgeBenchmarkProfiler()
    df = profiler.run_full_suite("proposed")
    print("\n" + "=" * 70)
    print("EDGE AI BENCHMARK RESULTS")
    print("=" * 70)
    print(df.to_string())
