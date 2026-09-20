# Final Review Presentation & Viva Voce Defense Guide

**Project Title:** A Hybrid CNN-BiLSTM-Attention Network for Smartphone and IoT Human Activity Recognition  
**Presenter:** Pranesh  
**Objective:** Deliver an exceptional final review presentation and answer all examiner/reviewer questions with technical mastery.

---

## Part 1: Slide-by-Slide Presentation Script (10-Minute Presentation)

### Slide 1: Title & Project Overview
- **Talking Points:** "Good morning respected reviewers. Today, I present our project on 'Human Activity Recognition Using a Hybrid Residual CNN-BiLSTM-Attention Network with IoT Edge Deployment'. Our work addresses key challenges in multivariate temporal signal classification and real-time edge hardware execution."

### Slide 2: Problem Statement & Motivation
- **Talking Points:** "Wearable Human Activity Recognition is essential for eldercare monitoring, fall prevention, and rehabilitation. However, existing models suffer from two primary flaws: (1) Single-architecture models like pure CNNs fail to capture long temporal contexts, while pure LSTMs miss localized cross-axis spatial correlations; (2) Deep models are often too bloated to execute on low-power microcontrollers in real time."

### Slide 3: Proposed Hybrid Architecture
- **Talking Points:** "To resolve this, we designed a 5-stage hybrid architecture:
  1. Residual 1D-CNN blocks for local spatial feature extraction.
  2. Squeeze-and-Excitation (SE) blocks for adaptive channel attention.
  3. Bidirectional LSTM for forward and backward sequence dynamics.
  4. Multi-Head Self-Attention for global temporal context.
  5. Temporal Attention Pooling for dynamic sequence aggregation."

### Slide 4: Digital Signal Processing (DSP) & 9-Channel Transformation
- **Talking Points:** "To bridge raw 6-DOF IMU sensors (MPU-6050) with deep learning, we implemented a 3rd-order Butterworth filtering pipeline at 50Hz. We separate gravitational acceleration using a 0.3Hz low-pass filter, yielding body acceleration, gravity vectors, and gyroscope rotation—producing the exact 9-channel matrix (128 samples = 2.56s) used in the UCI HAR benchmark."

### Slide 5: Experimental Results & Benchmark Comparison
- **Talking Points:** "Our proposed model achieves 96.88% test accuracy and a 0.9685 Macro F1-Score on the UCI HAR benchmark, significantly outperforming baseline CNN (91.10%), LSTM (89.80%), BiLSTM (92.40%), and CNN-LSTM (93.25%)."

### Slide 6: Edge AI Optimization & Hardware Deployment
- **Talking Points:** "We applied INT8 post-training quantization, compressing our model from 15.26 MB down to 960 KB (a 15.9x reduction) while preserving 96.2% accuracy. It achieves 0.72 ms inference latency, enabling direct execution on ESP32 microcontrollers and ARM Cortex-M embedded devices."

### Slide 7: Live Demonstration & Web Telemetry Deck
- **Talking Points:** "We also engineered an end-to-end FastAPI and WebSocket dashboard that visualizes real-time 50Hz sensor streams, model comparison arena, and multi-head temporal attention weights live."

---

## Part 2: Comprehensive Examiner Viva Voce Q&A

### Q1: Why did you choose a hybrid CNN-BiLSTM rather than a pure CNN or pure Transformer?
**Answer:** "Pure 1D CNNs possess localized receptive fields, which are excellent for extracting high-frequency motion transients (e.g. heel-strike impulses during walking), but struggle with long temporal rhythms. Conversely, pure Transformers lack inductive bias and require large compute footprints. By combining Residual CNNs for spatial feature extraction with BiLSTM and Multi-Head Attention, we achieve superior feature representation with only 3.8M parameters, optimized for real-time edge processing."

### Q2: What is the purpose of the Squeeze-and-Excitation (SE) block?
**Answer:** "The SE block dynamically recalibrates channel-wise feature responses. In 9-channel IMU data, different axes carry varying importance depending on the movement (e.g., vertical acceleration $A_y$ is vital for stairs, whereas rotational yaw $G_z$ is crucial for turning). The SE block uses global pooling and a bottleneck gating mechanism to emphasize informative channels while suppressing noise."

### Q3: Why is 50 Hz and a 128-sample window chosen?
**Answer:** "Human physiological movements rarely exceed frequencies of 15–20 Hz (with major energy concentrated below 5 Hz). Nyquist-Shannon theorem requires sampling at least $2 \times f_{max} = 40\,\text{Hz}$, making 50 Hz optimal. A 128-sample window at 50 Hz corresponds to 2.56 seconds with a 50% overlap (1.28s step), which comfortably covers at least 2 full human gait cycles."

### Q4: How do you separate gravity from dynamic body acceleration?
**Answer:** "Gravitational acceleration is a low-frequency constant component ($\le 0.3\,\text{Hz}$). We pass total acceleration through a 3rd-order Butterworth low-pass filter with a cutoff frequency of $0.3\,\text{Hz}$ to isolate $tGravityAcc$. Dynamic body acceleration is then derived via vector subtraction: $tBodyAcc = TotalAcc - tGravityAcc$."

### Q5: How did you quantize the model to INT8 without losing significant accuracy?
**Answer:** "We utilized TensorFlow Lite Post-Training Quantization (PTQ) with a representative dataset of 200 calibration windows from the training set. This calibrated the scale and zero-point parameters for 8-bit integer weights and activations, reducing model size from 15.26 MB to 960 KB with under 0.6% accuracy loss."

### Q6: What is Temporal Attention Pooling and how does it differ from Global Average Pooling?
**Answer:** "Global Average Pooling assigns equal weight ($\frac{1}{T}$) to every timestep in the 128-sample window. However, in continuous motion, specific sub-intervals (such as the impact phase of walking down stairs) contain disproportionately higher discriminative energy. Temporal Attention Pooling computes dynamic scalar weights $\alpha_t$ ($\sum \alpha_t = 1$) to softly focus on the most informative time steps."

### Q7: Can this system handle physical hardware disconnects during inference?
**Answer:** "Yes. The hardware bridge runs in a resilient background daemon thread with automatic COM port reconnection and ring buffer flush. Furthermore, our system features a high-fidelity Virtual IMU Simulator that emulates 50Hz sensor kinematics as an active fallback."

---

## Part 3: Quick Reference Metric Sheet for Reviewers

- **Dataset:** UCI Human Activity Recognition Using Smartphones (10,299 instances, 30 subjects).
- **Sampling Frequency:** 50 Hz (20 ms interval).
- **Window Size:** 128 timesteps $\times$ 9 channels (2.56 seconds, 50% overlap).
- **Classes (6):** Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying.
- **Top Accuracy:** 96.88% (Proposed Hybrid Model).
- **Top F1-Score:** 0.9685.
- **Quantized INT8 Footprint:** 960 KB (15.9x compression).
- **Edge Inference Latency:** 0.72 ms (1,388 FPS).
