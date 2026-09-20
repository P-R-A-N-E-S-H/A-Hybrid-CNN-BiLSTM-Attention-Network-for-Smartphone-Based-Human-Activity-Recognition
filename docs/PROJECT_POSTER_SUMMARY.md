# Project Exhibition Poster & Final Review Summary Sheet

## Project Title
**A Hybrid Residual CNN-BiLSTM-Attention Network for Real-Time Smartphone and IoT Human Activity Recognition**

**Author:** Pranesh  
**Domain:** Artificial Intelligence, Deep Learning, Edge IoT, Digital Signal Processing  

---

### 1. Problem Statement
Accurate continuous human activity recognition (HAR) on low-power wearable devices is limited by:
- Inability of pure 1D CNNs to capture long temporal contexts.
- Inability of standard LSTMs to extract cross-channel spatial-axis features.
- High memory and compute overhead of uncompressed deep models on microcontrollers.

---

### 2. Proposed Solution
1. **Multi-Scale Residual 1D CNNs**: High-frequency spatial motion transient extraction.
2. **Squeeze-and-Excitation (SE) Channel Attention**: Inter-axis dynamic weighting.
3. **Bidirectional LSTM**: Past and future sequence dynamics.
4. **Multi-Head Self-Attention**: Long-range temporal correlation.
5. **Temporal Attention Pooling**: Dynamic timestep importance weighting.
6. **Embedded Hardware Integration**: ESP32 / MPU-6050 50Hz telemetry + Butterworth 0.3Hz gravity separation + TFLite INT8 quantization.

---

### 3. Key Achievements & Metrics
- 🏆 **Test Accuracy:** **96.88%** (State of the art on UCI HAR benchmark).
- 🎯 **Macro F1-Score:** **0.9685**.
- ⚡ **Quantized INT8 Model Size:** **960 KB** (15.9x compression vs 15.26 MB baseline).
- ⏱️ **Edge Inference Latency:** **0.72 ms** (1,388 FPS throughput).
- 📟 **Real-time 50Hz Hardware Telemetry:** ESP32 + MPU6050 USB-Serial / BLE.
- 🌐 **Interactive Dashboard:** Live waveform visualizer, attention heatmaps, probability gauges.
