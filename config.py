"""
======================================================================
config.py - Centralized Configuration Module
======================================================================
Human Activity Recognition (HAR) System Configuration
Defines hyperparameters, DSP filter constants, hardware parameters,
model paths, and class labels.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed" / "har"
MODELS_DIR = BASE_DIR / "models"
TRAINED_MODELS_DIR = MODELS_DIR / "trained"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
DOCS_DIR = BASE_DIR / "docs"
HARDWARE_DIR = BASE_DIR / "hardware"
EDGE_DIR = BASE_DIR / "edge_deployment"

# Ensure directories exist
for directory in [RESULTS_DIR, FIGURES_DIR, DOCS_DIR, HARDWARE_DIR, EDGE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset & Signal Parameters
SAMPLING_RATE_HZ = 50          # 50 Hz sampling frequency (UCI HAR standard)
WINDOW_SIZE = 128              # 128 samples = 2.56 seconds window
WINDOW_STEP = 64               # 50% overlap = 64 samples step
NUM_CHANNELS = 9               # 9 channels: 3 tBodyAcc, 3 tGravityAcc, 3 tBodyGyro
NUM_CLASSES = 6                # 6 human activity categories

# Activity Classes Taxonomy (Matching models.py and class_names.npy)
ACTIVITY_LABELS = {
    0: "LAYING",
    1: "SITTING",
    2: "STANDING",
    3: "WALKING",
    4: "WALKING_DOWNSTAIRS",
    5: "WALKING_UPSTAIRS"
}

ACTIVITY_NAMES = [
    "LAYING",
    "SITTING",
    "STANDING",
    "WALKING",
    "WALKING_DOWNSTAIRS",
    "WALKING_UPSTAIRS"
]

ACTIVITY_COLORS = {
    "LAYING": "#6B7280",             # Slate Gray
    "SITTING": "#8B5CF6",            # Purple
    "STANDING": "#EC4899",           # Pink
    "WALKING": "#3B82F6",            # Blue
    "WALKING_DOWNSTAIRS": "#F59E0B", # Amber
    "WALKING_UPSTAIRS": "#10B981"    # Emerald
}

# Channel Descriptions (UCI HAR standard)
CHANNEL_NAMES = [
    "tBodyAcc-X", "tBodyAcc-Y", "tBodyAcc-Z",
    "tGravityAcc-X", "tGravityAcc-Y", "tGravityAcc-Z",
    "tBodyGyro-X", "tBodyGyro-Y", "tBodyGyro-Z"
]

# DSP Filtering Parameters
BUTTERWORTH_ORDER = 3
LOWPASS_CUTOFF_HZ = 20.0       # Noise reduction cutoff
GRAVITY_CUTOFF_HZ = 0.3        # Gravity separation cutoff (0.3 Hz)

# Model Training Hyperparameters
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
L2_REGULARIZATION = 1e-4
DROPOUT_RATE = 0.3
PATIENCE_EARLY_STOPPING = 15
PATIENCE_REDUCE_LR = 7

# Hardware Communication Parameters
SERIAL_BAUD_RATE = 115200
DEFAULT_SERIAL_PORT = "COM3"
SENSOR_I2C_ADDR = 0x68         # MPU-6050 standard I2C address
ACCEL_SCALE_FACTOR = 16384.0   # +/- 2g range (LSB/g)
GYRO_SCALE_FACTOR = 131.0      # +/- 250 deg/s range (LSB/(deg/s))

# Server & Dashboard Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
WS_ENDPOINT = "/ws/stream"
