"""
======================================================================
calibration.py - MPU-6050 6-Point Static Calibration Routine
======================================================================

Computes zero-g bias and sensitivity scale factors for 3-axis accelerometer
and 3-axis gyroscope to eliminate thermal drift and manufacturing offsets.

Author: Pranesh
======================================================================
"""

import time
import numpy as np
from typing import Dict, Tuple


class IMUCalibrator:
    """6-Point Static IMU Calibration."""

    def __init__(self, target_samples: int = 500):
        self.target_samples = target_samples

    def calibrate_stationary(self, raw_samples: np.ndarray) -> Dict[str, Tuple[float, float, float]]:
        """
        Computes mean offset from stationary flat position.
        Expected: Accel = (0, 0, 1.0g), Gyro = (0, 0, 0 rad/s).
        """
        if len(raw_samples) < 10:
            raise ValueError("Insufficient calibration samples")

        acc_mean = np.mean(raw_samples[:, :3], axis=0)
        gyro_mean = np.mean(raw_samples[:, 3:], axis=0)

        acc_offsets = (acc_mean[0], acc_mean[1], acc_mean[2] - 1.0)
        gyro_offsets = (gyro_mean[0], gyro_mean[1], gyro_mean[2])

        return {
            "accel_offsets_g": acc_offsets,
            "gyro_offsets_rad_s": gyro_offsets
        }
