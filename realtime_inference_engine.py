"""
======================================================================
realtime_inference_engine.py - Unified Real-Time Inference Pipeline
======================================================================

Multi-backend real-time inference manager supporting:
  1. Keras (.keras) models (Proposed Hybrid Attention, CNN, LSTM, BiLSTM, CNN-LSTM)
  2. TensorFlow Lite (.tflite) Quantized Interpreters
  3. Live Attention Weights Extraction
  4. Temporal confidence smoothing & state transition debounce logic
  5. Fall / rapid posture shift alert triggers

Author: Pranesh
======================================================================
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

import config
from models import TemporalAttentionPooling


class RealtimeInferenceEngine:
    """
    High-speed, multi-backend inference engine for continuous HAR telemetry.
    """

    def __init__(
        self,
        model_name: str = "proposed",
        backend: str = "keras",  # "keras" or "tflite"
        confidence_threshold: float = 0.65,
        smoothing_window: int = 3
    ):
        self.model_name = model_name
        self.backend = backend
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window

        self.custom_objects = {"TemporalAttentionPooling": TemporalAttentionPooling}
        self.model = None
        self.tflite_interpreter = None
        self.input_details = None
        self.output_details = None

        # Attention extractor sub-model if supported
        self.attention_model = None

        # State tracking
        self.history_probs: List[np.ndarray] = []
        self.last_activity = "UNKNOWN"
        self.last_confidence = 0.0
        self.activity_start_time = time.time()
        self.inference_count = 0
        self.total_latency_ms = 0.0

        self.load_model(model_name, backend)

    def load_model(self, model_name: str, backend: str = "keras"):
        """Loads model into memory."""
        self.model_name = model_name
        self.backend = backend

        if backend == "keras":
            model_file = os.path.join(config.TRAINED_MODELS_DIR, f"{model_name}.keras")
            if not os.path.exists(model_file):
                raise FileNotFoundError(f"Model file not found: {model_file}")

            print(f"[InferenceEngine] Loading Keras model: {model_file}...")
            self.model = tf.keras.models.load_model(model_file, custom_objects=self.custom_objects)
            self._setup_attention_extractor()
            print(f"[InferenceEngine] Successfully loaded {model_name} (Keras)")

        elif backend == "tflite":
            tflite_dir = os.path.join(config.BASE_DIR, "edge_deployment", "tflite_models")
            tflite_file = os.path.join(tflite_dir, f"{model_name}_fp32.tflite")
            if not os.path.exists(tflite_file):
                # Fallback to int8 or search any tflite
                tflite_file = os.path.join(tflite_dir, f"{model_name}_int8.tflite")
            if not os.path.exists(tflite_file):
                raise FileNotFoundError(f"TFLite model not found in {tflite_dir}")

            print(f"[InferenceEngine] Loading TFLite model: {tflite_file}...")
            self.tflite_interpreter = tf.lite.Interpreter(model_path=tflite_file)
            self.tflite_interpreter.allocate_tensors()
            self.input_details = self.tflite_interpreter.get_input_details()
            self.output_details = self.tflite_interpreter.get_output_details()
            print(f"[InferenceEngine] Successfully loaded {model_name} (TFLite)")

    def _setup_attention_extractor(self):
        """Constructs sub-model to extract intermediate temporal attention weights."""
        if self.model is None:
            return
        try:
            # Look for attention layer output in proposed model
            for layer in self.model.layers:
                if "attention" in layer.name.lower() or "dense" in layer.name.lower():
                    # Check if output has temporal dimension (None, 128, ...)
                    if len(layer.output.shape) == 3 and layer.output.shape[1] == config.WINDOW_SIZE:
                        self.attention_model = tf.keras.Model(
                            inputs=self.model.inputs,
                            outputs=layer.output
                        )
                        break
        except Exception as e:
            self.attention_model = None

    def predict_window(self, window_data: np.ndarray) -> Dict[str, Any]:
        """
        Runs inference on a single 9-channel window (128, 9).
        
        Args:
            window_data: (128, 9) normalized numpy array
        Returns:
            Dictionary with prediction results, probabilities, confidence,
            attention weights, and latency.
        """
        if window_data.ndim == 2:
            input_tensor = np.expand_dims(window_data, axis=0).astype(np.float32)
        else:
            input_tensor = window_data.astype(np.float32)

        t_start = time.perf_counter()

        if self.backend == "keras":
            raw_preds = self.model(input_tensor, training=False).numpy()[0]
        else:
            self.tflite_interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
            self.tflite_interpreter.invoke()
            raw_preds = self.tflite_interpreter.get_tensor(self.output_details[0]['index'])[0]

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        self.inference_count += 1
        self.total_latency_ms += latency_ms

        # Temporal smoothing
        self.history_probs.append(raw_preds)
        if len(self.history_probs) > self.smoothing_window:
            self.history_probs.pop(0)

        smoothed_probs = np.mean(self.history_probs, axis=0)
        pred_class_idx = int(np.argmax(smoothed_probs))
        confidence = float(smoothed_probs[pred_class_idx])
        predicted_label = config.ACTIVITY_NAMES[pred_class_idx]

        # Activity duration tracking
        now = time.time()
        if predicted_label != self.last_activity:
            self.last_activity = predicted_label
            self.activity_start_time = now

        duration_sec = round(now - self.activity_start_time, 1)

        # Extract temporal attention weights if available
        attention_weights = []
        if self.attention_model and self.backend == "keras":
            try:
                attn_out = self.attention_model(input_tensor, training=False).numpy()[0]
                # Average across feature dimension -> (128,)
                attn_1d = np.mean(np.abs(attn_out), axis=-1)
                # Normalize 0-1
                attn_norm = (attn_1d - np.min(attn_1d)) / (np.max(attn_1d) - np.min(attn_1d) + 1e-8)
                attention_weights = attn_norm.tolist()
            except Exception:
                pass

        return {
            "activity": predicted_label,
            "class_id": pred_class_idx,
            "confidence": round(confidence, 4),
            "confidence_percent": round(confidence * 100.0, 1),
            "probabilities": {
                name: round(float(smoothed_probs[i]), 4)
                for i, name in enumerate(config.ACTIVITY_NAMES)
            },
            "activity_duration_sec": duration_sec,
            "latency_ms": round(latency_ms, 2),
            "average_latency_ms": round(self.total_latency_ms / self.inference_count, 2),
            "backend": self.backend,
            "model_name": self.model_name,
            "attention_weights": attention_weights,
            "timestamp": now
        }


if __name__ == "__main__":
    print("Testing RealtimeInferenceEngine...")
    engine = RealtimeInferenceEngine(model_name="proposed", backend="keras")
    dummy = np.random.normal(0, 1, (128, 9)).astype(np.float32)
    res = engine.predict_window(dummy)
    print("Result:", res)
