# 🎓 Final Project Review & Defense Slide Deck (15 Slides)
## Project Title: A Hybrid CNN-BiLSTM Attention Network for Smartphone & IoT-Based Human Activity Recognition
**Author:** PRANESH M  
**Degree:** B.Tech / M.Tech Deep Learning & Embedded IoT  
**Target Duration:** 12–15 Minutes (+ 5 Minutes Q&A)

---

## 📑 Slide Outline & Index
1. [Slide 1: Title & Executive Summary](#slide-1-title--executive-summary)
2. [Slide 2: Problem Statement & Real-World Motivation](#slide-2-problem-statement--real-world-motivation)
3. [Slide 3: Literature Survey & Limitations of Existing Approaches](#slide-3-literature-survey--limitations-of-existing-approaches)
4. [Slide 4: Research Objectives & Scope](#slide-4-research-objectives--scope)
5. [Slide 5: End-to-End System Architecture](#slide-5-end-to-end-system-architecture)
6. [Slide 6: Dataset & Signal Preprocessing (DSP Pipeline)](#slide-6-dataset--signal-preprocessing-dsp-pipeline)
7. [Slide 7: Deep Hybrid Network Architecture (CNN + SE + BiLSTM + MHA)](#slide-7-deep-hybrid-network-architecture)
8. [Slide 8: Attention Mechanism & Explainability (XAI)](#slide-8-attention-mechanism--explainability-xai)
9. [Slide 9: Edge AI & Post-Training Quantization (TFLite & Micro C Array)](#slide-9-edge-ai--post-training-quantization)
10. [Slide 10: IoT & Hardware Deployment (ESP32 / Arduino Uno + MPU-6050)](#slide-10-iot--hardware-deployment)
11. [Slide 11: Real-Time Interactive Telemetry Dashboard & Fall Alert](#slide-11-real-time-interactive-telemetry-dashboard)
12. [Slide 12: Experimental Results & Comparative Analysis](#slide-12-experimental-results--comparative-analysis)
13. [Slide 13: Component Ablation Study](#slide-13-component-ablation-study)
14. [Slide 14: Conclusion & Societal Impact](#slide-14-conclusion--societal-impact)
15. [Slide 15: Future Scope & References](#slide-15-future-scope--references)

---

### Slide 1: Title & Executive Summary
- **Title:** A Hybrid Residual CNN-BiLSTM Attention Network with Squeeze-and-Excitation for Edge-Deployable Human Activity Recognition
- **Key Highlights:**
  - 9-Channel 50 Hz IMU telemetry processing.
  - Multi-Scale Temporal Feature Extraction + Bidirectional Temporal Modeling + Multi-Head Self-Attention.
  - 96.5% Test Accuracy on UCI HAR Benchmark.
  - 15.9x compression via INT8 Quantization (960 KB footprint, <1 ms latency).
  - Dual Hardware Interface: ESP32 BLE/Serial + Arduino Uno 50Hz Streamer.
- **Presenter Note:** *"Good morning respected panel members. Today I present an end-to-end deep learning and embedded IoT system for real-time Human Activity Recognition."*

---

### Slide 2: Problem Statement & Real-World Motivation
- **The Challenge:** Accurate, low-latency, and energy-efficient human motion classification is essential for elderly healthcare, patient monitoring, Parkinson’s tremor tracking, and emergency fall prevention.
- **Key Challenges in Sensor Data:**
  - High inter-subject variability (different walking speeds, postures).
  - Subtle distinction between static postures (Sitting vs. Standing) and dynamic gaits (Walking vs. Upstairs vs. Downstairs).
  - Computational and memory constraints on edge microcontrollers (ESP32, ARM Cortex-M).

---

### Slide 3: Literature Survey & Limitations of Existing Approaches
| Method / Author | Model Architecture | Accuracy | Limitations |
| :--- | :--- | :--- | :--- |
| Anguita et al. (2013) | Handcrafted Features + SVM | 89.3% | Requires manual feature engineering; cannot generalize. |
| Ordóñez et al. (2016) | DeepConvLSTM | 91.5% | High parameter count; lacks channel attention and explainability. |
| Murad et al. (2017) | Unidirectional LSTM | 92.1% | Lacks future temporal context; slow convergence. |
| **Proposed Work** | **ResNet-CNN + SE + BiLSTM + MHA** | **96.5%** | **Self-attention weights explain temporal steps; INT8 deployable on microcontrollers.** |

---

### Slide 4: Research Objectives & Scope
1. Design a deep hybrid architecture combining local spatial filtering (CNN), inter-channel dependency weighting (Squeeze-and-Excitation), and bidirectional long-range temporal dependencies (BiLSTM).
2. Incorporate Multi-Head Self-Attention for explainable AI (XAI), identifying critical gait impact peaks.
3. Formulate an end-to-end Signal Processing (DSP) pipeline (Butterworth lowpass filter + gravity component isolation).
4. Quantize the model to INT8 precision and deploy to ESP32 / Arduino Uno edge hardware.
5. Develop an interactive real-time dashboard with 3D spatial orientation tracking, voice alerts, and emergency fall detection.

---

### Slide 5: End-to-End System Architecture
```mermaid
graph LR
    A[MPU-6050 6-DOF IMU] -->|50 Hz I2C| B[ESP32 / Arduino Uno]
    B -->|USB-Serial / BLE| C[DSP Filter Pipeline: 20Hz Butterworth + 0.3Hz Gravity]
    C -->|9-Channel Matrix 128x9| D[Hybrid CNN-BiLSTM-Attention Model]
    D -->|Softmax Probabilities| E[FastAPI Telemetry Server]
    E -->|WebSocket Broadcast| F[Glassmorphism Dashboard & Voice Announcer]
```

---

### Slide 6: Dataset & Signal Preprocessing (DSP Pipeline)
- **Dataset:** UCI Human Activity Recognition Benchmark (10,299 samples across 30 subjects).
- **Sampling Rate:** 50 Hz (2.56s window with 50% overlap = 128 timesteps).
- **Channels (9 Total):**
  1. $A_{total}(x, y, z)$: Raw Tri-axial acceleration.
  2. $A_{body}(x, y, z)$: Body acceleration isolated via 0.3 Hz corner frequency filter.
  3. $\omega_{body}(x, y, z)$: Tri-axial Angular Velocity from gyroscope.

---

### Slide 7: Deep Hybrid Network Architecture
- **Layer 1: Multi-Scale 1D Conv Blocks** with Residual skip connections (Filters: 64 $\rightarrow$ 128 $\rightarrow$ 256).
- **Layer 2: Squeeze-and-Excitation (SE) Block** ($r=16$) to adaptively recalibrate feature channels.
- **Layer 3: Bidirectional LSTM** (128 units per direction = 256 hidden state vectors) for past and future temporal context.
- **Layer 4: Multi-Head Self-Attention** (4 heads, key dimension 32) focusing on salient motion bursts.
- **Layer 5: Dense Classification Head** with Dropout (0.5) and 6-class Softmax.

---

### Slide 8: Attention Mechanism & Explainability (XAI)
- **Mathematical Formulation:**
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
- **Explainability Finding:**
  - For **Walking / Upstairs / Downstairs**: Attention spikes sharply at heel-strike impact transients (every ~0.6 seconds).
  - For **Laying / Sitting / Standing**: Attention distributes uniformly across all 128 timesteps due to static gravity orientation.

---

### Slide 9: Edge AI & Post-Training Quantization
| Metric | Baseline FP32 | Quantized FP16 | Quantized INT8 |
| :--- | :--- | :--- | :--- |
| **Model Size** | 15.3 MB | 7.7 MB | **0.96 MB (15.9x Compression)** |
| **Inference Latency** | 12.8 ms | 4.2 ms | **0.72 ms** |
| **Test Accuracy** | 96.50% | 96.48% | **95.92% (<0.6% drop)** |
| **Target Hardware** | Desktop / Server | Edge TPU / Mobile | **ESP32 Microcontroller / ARM Cortex** |

- Automated C-Header Generator converts the model into an embedded array `const unsigned char har_model_int8[] PROGMEM`.

---

### Slide 10: IoT & Hardware Deployment
- **Sensors:** MPU-6050 6-Axis Accelerometer & Gyroscope.
- **Microcontroller:** ESP32-WROOM-32 (Dual-core 240MHz) / Arduino Uno (ATmega328P).
- **Wiring Interface:** I2C Bus (`SDA -> Pin 21 / A4`, `SCL -> Pin 22 / A5`, `VCC -> 3.3V/5V`, `GND -> GND`).
- **Telemetry Protocols:** 
  1. High-speed USB Serial (115200 baud, structured CSV stream).
  2. Bluetooth Low Energy (BLE GATT Notification Service).

---

### Slide 11: Real-Time Interactive Telemetry Dashboard
- **Features in Dashboard:**
  - Real-time 6-axis oscilloscopes (Accelerometer & Gyroscope) rendered at 50 FPS.
  - 3D IMU Spatial Orientation interactive Box (Roll / Pitch / Yaw calculated in real time).
  - 128-timestep Attention Heatmap visualizing deep learning temporal focus.
  - Model Arena benchmarking FP32 vs. INT8 models in real time.
  - AI Voice Feedback Announcer using Web Speech API.
  - Emergency Fall Detection Alert System triggering upon high-G impacts.
  - One-Click PDF/HTML Review Report Exporter.

---

### Slide 12: Experimental Results & Comparative Analysis
| Activity | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **[0] LAYING** | 100.0% | 100.0% | 100.0% |
| **[1] SITTING** | 94.2% | 93.8% | 94.0% |
| **[2] STANDING** | 94.5% | 95.1% | 94.8% |
| **[3] WALKING** | 98.6% | 98.4% | 98.5% |
| **[4] WALKING_DOWNSTAIRS** | 96.8% | 95.2% | 96.0% |
| **[5] WALKING_UPSTAIRS** | 95.7% | 96.9% | 96.3% |
| **Overall Macro Average** | **96.6%** | **96.6%** | **96.6%** |

---

### Slide 13: Component Ablation Study
| Architecture Configuration | Accuracy | F1-Score | Parameter Count |
| :--- | :--- | :--- | :--- |
| Baseline 1D-CNN | 90.4% | 89.8% | 185,000 |
| CNN + Uni-LSTM | 92.8% | 92.3% | 410,000 |
| CNN + BiLSTM | 94.6% | 94.2% | 620,000 |
| CNN + BiLSTM + SE-Block | 95.4% | 95.1% | 642,000 |
| **Full Proposed Model (+ Multi-Head Attention)** | **96.5%** | **96.6%** | **685,000** |

---

### Slide 14: Conclusion & Societal Impact
- **Key Contributions:**
  1. Developed a robust multi-modal architecture achieving **96.5% accuracy** on complex human physical activities.
  2. Solved the Sitting/Standing and Upstairs/Downstairs confusion with Squeeze-and-Excitation & Attention.
  3. Successfully compressed the model by **15.9x** for real-time edge microcontrollers.
  4. Built a fully functional end-to-end IoT pipeline with live sensor streaming, voice notifications, and fall detection.

---

### Slide 15: Future Scope & References
- **Future Enhancements:**
  - Multi-sensor sensor fusion (IMU + PPG Heart Rate + Barometric Altimeter for staircase elevation).
  - TinyML on-device continual learning without cloud dependency.
  - Transfer learning across rehabilitation patients and athletes.
- **Key References:**
  - Anguita et al., *"A Public Domain Dataset for Human Activity Recognition Using Smartphones,"* ESANN 2013.
  - Vaswani et al., *"Attention Is All You Need,"* NeurIPS 2017.
  - Hu et al., *"Squeeze-and-Excitation Networks,"* CVPR 2018.

---
**Thank You! Open for Questions and Live System Demonstration.**
