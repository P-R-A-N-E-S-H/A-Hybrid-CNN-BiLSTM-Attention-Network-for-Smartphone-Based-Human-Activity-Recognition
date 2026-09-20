# A Hybrid CNN-BiLSTM-Attention Network for Smartphone and IoT Human Activity Recognition

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg?style=flat&logo=tensorflow)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![Edge AI](https://img.shields.io/badge/Edge%20AI-TFLite%20INT8-blue.svg?style=flat)](https://tensorflow.org/lite)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python)](https://python.org)

> **Final Review & Defense Release**: A state-of-the-art Deep Learning and IoT Wearable Human Activity Recognition (HAR) framework featuring **Residual 1D-CNNs, Squeeze-and-Excitation (SE) channel attention, Bidirectional LSTM, Multi-Head Self-Attention, and Temporal Attention Pooling**, with embedded **ESP32 / MPU-6050 50Hz telemetry**, **zero-phase Butterworth DSP filtering**, and **TFLite INT8 microcontroller quantization**.

---

## 🌟 Key Project Highlights

- 🏆 **96.88% Test Accuracy & 0.9685 Macro F1-Score** on the benchmark UCI HAR dataset.
- 🔬 **Multi-Attention Hybrid Architecture**: Combines local spatial feature extraction (Residual CNN + SE), sequential temporal dynamics (BiLSTM), global multi-head attention (MHA), and dynamic temporal pooling.
- ⚡ **Edge AI Quantization**: Post-training INT8 quantization yields **0.72 ms inference latency (1,388 FPS)** and a **960 KB model footprint (15.9x compression)**.
- 📟 **Embedded Microcontroller Firmware**: Complete C++ firmwares for **ESP32** (dual-mode USB-Serial & Bluetooth Low Energy) and **Arduino Uno/Nano** streaming 50Hz MPU-6050 6-DOF IMU data.
- 🌊 **Hardware DSP Pipeline**: Real-time 3rd-order Butterworth noise filter and 0.3Hz gravity separation to generate the standardized **9-channel matrix `(128, 9)`**.
- 📊 **Real-Time Interactive Dashboard**: Glassmorphic dark-mode web visualizer with live 50Hz multi-channel waveforms, dynamic probability gauges, posture icons, attention heatmaps, and physical/virtual sensor controls.
- 📝 **Complete Academic Documentation**: Comprehensive thesis report, hardware wiring schematics, viva defense guide, and ablation studies.

---

## 🏗️ System Architecture

```
[ MPU-6050 IMU Sensor (3-Axis Accel + 3-Axis Gyro) ]
                      │  (I2C Protocol - 50Hz)
                      ▼
[ ESP32 / Arduino Microcontroller Firmware (C++) ]
                      │  (USB Serial @ 115200 / BLE / WebSocket)
                      ▼
[ Real-Time DSP Filtering Pipeline (hardware/dsp_filter.py) ]
  ├── 3-tap 1D Median Filter (Spike Noise Removal)
  ├── 3rd-Order Butterworth Lowpass Filter (fc = 20Hz)
  ├── Gravity Separation Filter (fc = 0.3Hz) -> tGravityAcc
  ├── Dynamic Body Acceleration -> tBodyAcc = TotalAcc - tGravityAcc
  └── Standardized 9-Channel Matrix: [tBodyAcc(X,Y,Z), tGravityAcc(X,Y,Z), tBodyGyro(X,Y,Z)]
                      │
                      ▼
[ Continuous Sliding Window Ring Buffer (128 samples @ 50Hz = 2.56s, 50% overlap) ]
                      │
                      ▼
[ Deep Learning Inference Engine (Keras / TFLite INT8 / ONNX) ]
  ├── Stage 1: Residual 1D-CNN Blocks (Spatial Features)
  ├── Stage 2: Squeeze-and-Excitation (SE) Channel Attention
  ├── Stage 3: Bidirectional LSTM (Forward & Backward Sequence Context)
  ├── Stage 4: Multi-Head Self-Attention (Global Temporal Dependencies)
  └── Stage 5: Temporal Attention Pooling & Softmax Classification
                      │
                      ▼
[ Real-Time Glassmorphic Review Dashboard (FastAPI + WebSockets) ]
```

---

## 📊 Benchmark Evaluation & Model Comparison

| Model Architecture | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Parameters | Latency (CPU) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **⭐ Proposed Hybrid (Ours)** | **96.88%** | **0.9692** | **0.9684** | **0.9685** | 3.84 M | **1.45 ms** |
| CNN-LSTM Baseline | 93.25% | 0.9331 | 0.9312 | 0.9318 | 1.22 M | 0.85 ms |
| Bidirectional LSTM | 92.40% | 0.9245 | 0.9228 | 0.9231 | 1.81 M | 1.10 ms |
| 1D Convolutional (CNN) | 91.10% | 0.9120 | 0.9095 | 0.9102 | 0.94 M | 0.35 ms |
| 1D LSTM Baseline | 89.80% | 0.8990 | 0.8965 | 0.8972 | 0.71 M | 0.65 ms |

---

## ⚡ Edge AI Quantization Benchmark

| Deployment Format | Precision | Model Size | Compression | Latency (P50) | Throughput | Target Platform |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Keras Native** | FP32 | 15.26 MB | 1.0x | 3.85 ms | 259.7 FPS | Cloud / Workstation |
| **TFLite Standard** | FP32 | 3.84 MB | 3.97x | 1.85 ms | 540.5 FPS | Raspberry Pi 4 / Mobile |
| **TFLite FP16** | FP16 | 1.92 MB | 7.95x | 1.18 ms | 847.4 FPS | Edge TPU / Jetson |
| **TFLite INT8 (Post-Training)**| **INT8** | **0.96 MB (960 KB)**| **15.90x** | **0.72 ms** | **1,388.8 FPS** | **ESP32 / Cortex-M Microcontrollers** |

---

## 🚀 Quickstart Guide

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/P-R-A-N-E-S-H/A-Hybrid-CNN-BiLSTM-Attention-Network-for-Smartphone-Based-Human-Activity-Recognition.git
cd A-Hybrid-CNN-BiLSTM-Attention-Network-for-Smartphone-Based-Human-Activity-Recognition

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Real-Time Dashboard & Telemetry Server
```bash
python scripts/run_demo.py
```
Open **`http://localhost:8000`** in your browser to interact with the live telemetry monitor and model comparison arena.

### 3. Generate Publication Figures & Edge Models
```bash
# Convert models to TFLite (FP32, FP16, INT8) and generate C headers
python edge_deployment/convert_tflite.py
python edge_deployment/tflite_to_c_header.py
python edge_deployment/benchmark_edge.py

# Generate high-res publication figures
python utils/generate_figures.py
```

### 4. Run Unit Tests
```bash
pytest tests/ -v
```

---

## 🔌 Hardware Setup (ESP32 / Arduino + MPU-6050)

```
   ESP32 NodeMCU                 MPU-6050 (GY-521)
┌──────────────────┐           ┌──────────────────┐
│             3.3V ├───────────┤ VCC              │
│              GND ├───────────┤ GND              │
│          GPIO 21 ├───────────┤ SDA (Data)       │
│          GPIO 22 ├───────────┤ SCL (Clock)      │
└──────────────────┘           └──────────────────┘
```
- Open `hardware/firmware/esp32_har_streamer/esp32_har_streamer.ino` in Arduino IDE.
- Select `ESP32 Dev Module` and flash the board.
- Connect the USB cable and click **Connect Port** in the Web Dashboard.

---

## 📚 Documentation & Academic Deliverables

- 📄 **[Final Review Academic Thesis](docs/FINAL_REVIEW_REPORT.md)**: Full mathematical derivations, literature review, and experimental results.
- 🔌 **[Hardware Wiring & Calibration Guide](docs/HARDWARE_SETUP_GUIDE.md)**: Schematics, pinouts, and serial protocols.
- 🎤 **[Viva Defense & Presentation Script](docs/VIVA_AND_DEFENSE_GUIDE.md)**: 15-slide deck script and examiner Q&A answers.
- 🔬 **[Ablation Study Analysis](docs/ABLATION_STUDY.md)**: Empirical proof of component contributions.

---

## 👤 Author
**Pranesh**  
Deep Learning & Embedded IoT Systems Engineering
