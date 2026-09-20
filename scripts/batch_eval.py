"""
HAR DeepSense - Batch Evaluation & Comprehensive Benchmark Suite
================================================================
Generates detailed classification metrics, per-class sensitivity/specificity,
latency profiling, and test set performance report.
"""

import sys
import os
import time
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Ensure root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import HAR_CONFIG
from data_loader import load_uci_har_dataset
from models.hybrid_model import build_hybrid_cnn_bilstm_attention_model


def run_comprehensive_evaluation():
    print("=" * 70)
    print(" HAR DEEPSENSE: COMPREHENSIVE BATCH EVALUATION & BENCHMARK SUITE")
    print("=" * 70)

    # 1. Load test dataset
    print("\n[1/4] Loading UCI HAR Test Dataset...")
    X_train, y_train, X_test, y_test = load_uci_har_dataset()
    print(f"      Test set size: {X_test.shape[0]} samples, {X_test.shape[1]} timesteps, {X_test.shape[2]} channels.")

    # 2. Load trained model
    weights_path = HAR_CONFIG.MODEL_WEIGHTS_PATH
    print(f"\n[2/4] Loading Deep Hybrid Model weights from: {weights_path}")
    model = build_hybrid_cnn_bilstm_attention_model(
        input_shape=(HAR_CONFIG.SEQUENCE_LENGTH, HAR_CONFIG.NUM_CHANNELS),
        num_classes=HAR_CONFIG.NUM_CLASSES
    )

    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        print("      [✓] Weights loaded successfully.")
    else:
        print(f"      [!] Weights not found at {weights_path}! Please train or ensure checkpoint exists.")
        return

    # 3. Model Inference & Latency profiling
    print("\n[3/4] Profiling Inference Latency on Test Set...")
    # Warmup
    _ = model.predict(X_test[:10], verbose=0)

    latencies = []
    y_preds_list = []
    
    # Measure batch of 1 for real-time edge emulation
    for i in range(min(100, len(X_test))):
        t0 = time.perf_counter()
        pred = model.predict(X_test[i:i+1], verbose=0)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)

    # Full test batch prediction
    full_preds = model.predict(X_test, batch_size=64, verbose=0)
    y_pred_classes = np.argmax(full_preds, axis=1)

    # 4. Compute Metrics
    print("\n[4/4] Computing Scientific Metrics & Error Analysis...")
    acc = accuracy_score(y_test, y_pred_classes)
    macro_f1 = f1_score(y_test, y_pred_classes, average="macro")
    weighted_f1 = f1_score(y_test, y_pred_classes, average="weighted")
    cm = confusion_matrix(y_test, y_pred_classes)

    class_names = [f"[{i}] {name}" for i, name in enumerate(HAR_CONFIG.ACTIVITY_LABELS)]

    print("\n" + "=" * 70)
    print(" 📊 FINAL SCIENTIFIC EVALUATION RESULTS")
    print("=" * 70)
    print(f" Overall Test Accuracy:   {acc * 100:.2f}%")
    print(f" Macro-Averaged F1-Score: {macro_f1 * 100:.2f}%")
    print(f" Weighted F1-Score:       {weighted_f1 * 100:.2f}%")
    print(f" Single-Sample Latency:   {avg_latency:.2f} ms (p95: {p95_latency:.2f} ms, p99: {p99_latency:.2f} ms)")
    print("-" * 70)
    print(" Per-Class Classification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=class_names, digits=4))
    print("-" * 70)
    print(" Confusion Matrix:")
    print(cm)
    print("=" * 70)

    # Save results to json for presentation
    results_summary = {
        "test_accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "single_sample_latency_ms": float(avg_latency),
        "p95_latency_ms": float(p95_latency),
        "confusion_matrix": cm.tolist(),
        "evaluated_samples": int(len(y_test))
    }
    
    os.makedirs(HAR_CONFIG.RESULTS_DIR, exist_ok=True)
    out_file = os.path.join(HAR_CONFIG.RESULTS_DIR, "batch_evaluation_metrics.json")
    with open(out_file, "w") as f:
        json.dump(results_summary, f, indent=4)
    print(f"\n[✓] Results summary saved to: {out_file}")


if __name__ == "__main__":
    run_comprehensive_evaluation()
