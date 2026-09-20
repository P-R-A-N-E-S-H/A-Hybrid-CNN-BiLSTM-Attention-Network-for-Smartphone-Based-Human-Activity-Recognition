"""
======================================================================
dsp_filter.py - Digital Signal Processing & Feature Extraction Pipeline
======================================================================

Implements the exact mathematical signal processing pipeline used by the
UCI HAR benchmark to convert raw 6-DOF IMU streams (3-axis Accel + 3-axis Gyro)
into the standardized 9-channel continuous feature matrix:
  1-3: tBodyAcc-XYZ     (Body linear acceleration)
  4-6: tGravityAcc-XYZ  (Gravitational constant component)
  7-9: tBodyGyro-XYZ    (Angular velocity)

Pipeline:
  1. Median Filter: Removes spurious spike noise.
  2. 3rd-Order Butterworth Low-pass Filter (fc = 20 Hz): Suppresses high-frequency noise.
  3. Butterworth Low-pass Filter (fc = 0.3 Hz): Separates constant gravity acceleration.
  4. Body Acceleration Subtraction: tBodyAcc = TotalAcc - tGravityAcc.
  5. 128-Sample Sliding Window Buffer with 50% overlap (2.56s window @ 50 Hz).
  6. Standard Normalization (Z-score scaling).

Author: Pranesh
======================================================================
"""

import numpy as np
from scipy import signal
import joblib
import os
from pathlib import Path
from typing import Tuple, Optional, Union, List


class HARSignalProcessor:
    """
    Real-time Digital Signal Processing (DSP) and feature extraction engine
    for continuous IMU sensor streams.
    """

    def __init__(
        self,
        sampling_rate_hz: float = 50.0,
        window_size: int = 128,
        window_step: int = 64,
        noise_cutoff_hz: float = 20.0,
        gravity_cutoff_hz: float = 0.3,
        filter_order: int = 3,
        scaler_path: Optional[str] = None
    ):
        self.fs = sampling_rate_hz
        self.nyquist = 0.5 * self.fs
        self.window_size = window_size
        self.window_step = window_step
        self.noise_cutoff = noise_cutoff_hz
        self.gravity_cutoff = gravity_cutoff_hz
        self.filter_order = filter_order

        # Design Butterworth filters
        # 1. Lowpass noise filter (0 - 20 Hz)
        norm_noise_cutoff = self.noise_cutoff / self.nyquist
        if norm_noise_cutoff >= 1.0:
            norm_noise_cutoff = 0.99
        self.b_noise, self.a_noise = signal.butter(
            self.filter_order, norm_noise_cutoff, btype='low', analog=False
        )

        # 2. Lowpass gravity filter (0 - 0.3 Hz)
        norm_gravity_cutoff = self.gravity_cutoff / self.nyquist
        self.b_gravity, self.a_gravity = signal.butter(
            self.filter_order, norm_gravity_cutoff, btype='low', analog=False
        )

        # Ring buffer for streaming data (N, 6): [ax, ay, az, gx, gy, gz]
        self.raw_buffer = np.zeros((0, 6), dtype=np.float32)

        # Pre-fitted standard scaler
        self.scaler = None
        if scaler_path and os.path.exists(scaler_path):
            try:
                self.scaler = joblib.load(scaler_path)
            except Exception as e:
                print(f"[DSP Warning] Failed to load scaler from {scaler_path}: {e}")

    def apply_median_filter(self, data: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Applies 1D median filter along time axis for each channel."""
        if len(data) < kernel_size:
            return data
        filtered = np.zeros_like(data)
        for ch in range(data.shape[1]):
            filtered[:, ch] = signal.medfilt(data[:, ch], kernel_size=kernel_size)
        return filtered

    def apply_noise_filter(self, data: np.ndarray) -> np.ndarray:
        """Applies 3rd-order Butterworth low-pass filter (20Hz) to suppress noise."""
        if len(data) <= 15:  # Scipy filtfilt requires minimum length
            return data
        # Use filtfilt for zero-phase distortion
        filtered = np.zeros_like(data)
        for ch in range(data.shape[1]):
            filtered[:, ch] = signal.filtfilt(self.b_noise, self.a_noise, data[:, ch])
        return filtered

    def separate_gravity_and_body(
        self, total_accel: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Separates total acceleration into gravity and dynamic body acceleration.
        
        Args:
            total_accel: (N, 3) matrix of [acc_x, acc_y, acc_z]
        Returns:
            body_accel: (N, 3) dynamic body acceleration
            gravity_accel: (N, 3) low-frequency gravitational component
        """
        if len(total_accel) <= 15:
            # Fallback for short batches: mean as gravity
            gravity = np.tile(np.mean(total_accel, axis=0, keepdims=True), (len(total_accel), 1))
            body = total_accel - gravity
            return body, gravity

        gravity_accel = np.zeros_like(total_accel)
        for axis in range(3):
            gravity_accel[:, axis] = signal.filtfilt(
                self.b_gravity, self.a_gravity, total_accel[:, axis]
            )

        body_accel = total_accel - gravity_accel
        return body_accel, gravity_accel

    def process_raw_6axis_window(self, raw_window: np.ndarray) -> np.ndarray:
        """
        Processes a raw 6-axis window (128, 6) [Ax, Ay, Az, Gx, Gy, Gz]
        into a 9-channel UCI HAR standard matrix (128, 9).
        
        Channels:
          [0:3] = tBodyAcc-XYZ
          [3:6] = tGravityAcc-XYZ
          [6:9] = tBodyGyro-XYZ
        """
        if raw_window.shape != (self.window_size, 6):
            raise ValueError(f"Expected shape ({self.window_size}, 6), got {raw_window.shape}")

        # 1. Median Filtering
        cleaned = self.apply_median_filter(raw_window, kernel_size=3)

        # 2. 20Hz Noise Filtering
        accel_total = self.apply_noise_filter(cleaned[:, 0:3])
        gyro_body = self.apply_noise_filter(cleaned[:, 3:6])

        # 3. Gravity Separation
        body_accel, gravity_accel = self.separate_gravity_and_body(accel_total)

        # 4. Construct 9-channel representation: [total_acc, body_acc, body_gyro]
        nine_channel = np.hstack([accel_total, body_accel, gyro_body])  # (128, 9)

        # 5. Apply Normalization if scaler is available
        if self.scaler is not None:
            # Reshape for scaler (128 * 9 or standard 9-dim transform)
            original_shape = nine_channel.shape
            try:
                nine_channel = self.scaler.transform(nine_channel)
            except Exception:
                # If scaler was fitted on flattened window
                flat = nine_channel.reshape(1, -1)
                transformed = self.scaler.transform(flat)
                nine_channel = transformed.reshape(original_shape)

        return nine_channel.astype(np.float32)

    def add_streaming_samples(self, samples: np.ndarray) -> List[np.ndarray]:
        """
        Feeds incoming continuous samples into the internal ring buffer.
        Returns any newly completed (128, 9) processed windows.
        
        Args:
            samples: (K, 6) new incoming raw sensor readings.
        Returns:
            List of (128, 9) numpy arrays ready for model inference.
        """
        if samples.ndim == 1:
            samples = samples.reshape(1, 6)

        self.raw_buffer = np.vstack([self.raw_buffer, samples])
        ready_windows = []

        while len(self.raw_buffer) >= self.window_size:
            current_window = self.raw_buffer[:self.window_size, :]
            processed = self.process_raw_6axis_window(current_window)
            ready_windows.append(processed)
            # Slide buffer by step size (50% overlap = 64 samples)
            self.raw_buffer = self.raw_buffer[self.window_step:, :]

        return ready_windows

    def reset(self):
        """Clears the streaming ring buffer."""
        self.raw_buffer = np.zeros((0, 6), dtype=np.float32)
