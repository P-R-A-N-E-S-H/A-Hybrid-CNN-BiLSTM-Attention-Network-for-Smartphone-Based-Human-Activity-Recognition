"""
HAR DeepSense - Interactive Terminal CLI Inspector
===================================================
A real-time ASCII HUD for terminal-based sensor inspection,
gravity separation telemetry, and live activity inference.
"""

import sys
import os
import time
import argparse
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import HAR_CONFIG
from inference import RealTimeHARInferenceEngine
from hardware.virtual_hardware_simulator import VirtualKinematicsSimulator
from hardware.hardware_bridge import SerialHardwareBridge


ACTIVITIES = ["LAYING", "SITTING", "STANDING", "WALKING", "WALKING_DOWNSTAIRS", "WALKING_UPSTAIRS"]
ICONS = {
    "LAYING": "🛌",
    "SITTING": "🪑",
    "STANDING": "🧍",
    "WALKING": "🚶",
    "WALKING_DOWNSTAIRS": "⛷️",
    "WALKING_UPSTAIRS": "🧗"
}


def make_bar(prob, length=20):
    filled = int(round(prob * length))
    return "█" * filled + "░" * (length - filled)


def run_cli_inspector(source_type="virtual", port=None, baudrate=115200):
    print("\n" + "=" * 65)
    print(" 📡 HAR DEEPSENSE: REAL-TIME TERMINAL HUD & SENSOR INSPECTOR")
    print("=" * 65)
    print(f" Source Mode: {source_type.upper()}")
    print(" Initializing Inference Engine & DSP Filter Pipeline...")

    engine = RealTimeHARInferenceEngine()
    print(" [✓] Deep Learning Model & Normalizer Loaded.")

    current_sample = [0.0] * 6
    latest_prediction = None
    step_count = 0

    if source_type == "virtual":
        sim = VirtualKinematicsSimulator(sample_rate_hz=50.0)
        sim.set_activity("WALKING")
        sim.start()
        print(" [✓] Virtual Kinematics Simulator running at 50 Hz.")
    else:
        bridge = SerialHardwareBridge(port=port, baudrate=baudrate)
        if not bridge.connect():
            print(f" [!] Could not connect to serial port {port}. Fallback to virtual simulator.")
            sim = VirtualKinematicsSimulator(sample_rate_hz=50.0)
            sim.start()
        else:
            print(f" [✓] Hardware connected to {port} at {baudrate} baud.")

    print("\nStarting live telemetry stream. Press Ctrl+C to stop.\n")
    time.sleep(1.0)

    try:
        while True:
            # Fetch sample
            if source_type == "virtual":
                sample = sim.generate_sample()
            else:
                sample = bridge.get_latest_sample()
                if sample is None:
                    time.sleep(0.02)
                    continue

            # Push to inference engine
            pred = engine.push_sample(sample)
            if pred is not None:
                latest_prediction = pred

            step_count += 1
            if step_count % 10 == 0:  # Update terminal at 5 Hz
                # Clear terminal
                os.system("cls" if os.name == "nt" else "clear")

                ax, ay, az, gx, gy, gz = sample
                print("=" * 65)
                print(" 📡 HAR-DEEPSENSE PRO: LIVE SENSOR TELEMETRY & ATTENTION HUD")
                print("=" * 65)
                print(f" Acceleration (g):    X={ax:+6.3f} | Y={ay:+6.3f} | Z={az:+6.3f}")
                print(f" Gyroscope (rad/s):   X={gx:+6.3f} | Y={gy:+6.3f} | Z={gz:+6.3f}")
                print("-" * 65)

                if latest_prediction:
                    act = latest_prediction["activity"]
                    act_idx = latest_prediction["activity_index"]
                    conf = latest_prediction["confidence"] * 100.0
                    lat = latest_prediction["inference_latency_ms"]
                    probs = latest_prediction["probabilities"]
                    icon = ICONS.get(act, "🎯")

                    print(f" CLASSIFICATION: {icon} [{act_idx}] {act:<20} | Confidence: {conf:5.1f}% | Latency: {lat:.2f}ms")
                    print("-" * 65)
                    print(" Softmax Class Distribution:")
                    for idx, (label, p) in enumerate(probs.items()):
                        bar = make_bar(p, length=24)
                        marker = " 👈 [ACTIVE]" if label == act else ""
                        print(f"  [{idx}] {label:<20} |{bar}| {p*100:5.1f}%{marker}")
                    print("-" * 65)

                    if "attention_weights" in latest_prediction:
                        attn = latest_prediction["attention_weights"]
                        # Sample 8 bins of attention
                        bins = np.array_split(attn, 8)
                        avg_bins = [np.mean(b) for b in bins]
                        max_b = max(avg_bins) if max(avg_bins) > 0 else 1.0
                        attn_str = " ".join([f"{b/max_b*100:3.0f}%" for b in avg_bins])
                        print(f" Temporal Attention Profile (8 Segments): [{attn_str}]")
                else:
                    print(" Buffering initial 128 samples (2.56s window)...")

                print("=" * 65)
                print(" [Ctrl+C] to Exit | Real-Time 50Hz Sampling Active")

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n[!] Stopping telemetry inspector...")
        if source_type == "virtual":
            sim.stop()
        print("[✓] Cleaned up and disconnected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HAR DeepSense Terminal CLI Inspector")
    parser.add_argument("--source", choices=["virtual", "serial"], default="virtual", help="Telemetry data source")
    parser.add_argument("--port", type=str, default="COM11", help="Serial COM port for hardware")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    args = parser.parse_args()

    run_cli_inspector(source_type=args.source, port=args.port, baudrate=args.baud)
