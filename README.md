Yes. **Copy everything inside this single code block and paste it directly into `README.md` on GitHub.**

````markdown
# 🛡️ Wearable AI — Real-Time Human Activity & Fall Detection

<p align="center">

<img src="https://img.shields.io/badge/AI-Deep%20Learning-blue?style=for-the-badge">
<img src="https://img.shields.io/badge/ESP32-IoT-green?style=for-the-badge">
<img src="https://img.shields.io/badge/MPU6050-Sensor-orange?style=for-the-badge">
<img src="https://img.shields.io/badge/Android-Mobile-black?style=for-the-badge">
<img src="https://img.shields.io/badge/TensorFlow-Lite-red?style=for-the-badge">
<img src="https://img.shields.io/badge/BLE-Bluetooth-purple?style=for-the-badge">

</p>

<p align="center">
<b>AI-Powered Wearable System for Real-Time Human Activity Recognition and Fall Detection</b>
</p>

---

## 🚀 Project Overview

**Wearable AI** is an end-to-end Artificial Intelligence + IoT system designed to recognize human activities and detect potential falls using real-time inertial sensor data.

The system combines:

- 🧠 Deep Learning
- 📡 ESP32
- 📊 MPU6050 Accelerometer & Gyroscope
- 📱 Android Application
- 🔵 Bluetooth Low Energy (BLE)
- ⚡ TensorFlow Lite
- 📍 GPS
- 📩 Emergency SMS
- ☁️ Cloud Logging

The wearable device collects motion data using the **MPU6050**. The **ESP32** processes the sensor readings and transmits them to an Android smartphone through BLE.

The Android application performs preprocessing, feature engineering, sliding-window generation and deep-learning inference using a TensorFlow Lite model.

---

# 🎯 Objectives

- Develop a wearable Human Activity Recognition system.
- Detect potentially dangerous falls using inertial sensor data.
- Collect real-time accelerometer and gyroscope measurements.
- Perform time-series preprocessing.
- Engineer meaningful motion features.
- Generate fixed-size sliding windows.
- Train multiple deep-learning architectures.
- Compare model performance.
- Convert the selected model to TensorFlow Lite.
- Deploy the model on Android.
- Perform real-time activity prediction.
- Implement fall confirmation logic.
- Obtain GPS location after a confirmed fall.
- Send emergency notifications.
- Build a complete wearable AI prototype.

---

# 🏗️ System Architecture

```text
                         WEARABLE DEVICE

                    ┌─────────────────────┐
                    │       MPU6050       │
                    │                     │
                    │  Accelerometer      │
                    │  Gyroscope          │
                    └──────────┬──────────┘
                               │
                              I²C
                               │
                               ▼
                    ┌─────────────────────┐
                    │        ESP32        │
                    │                     │
                    │ Sensor Acquisition  │
                    │ Sampling @ 50 Hz    │
                    │ BLE Communication   │
                    └──────────┬──────────┘
                               │
                              BLE
                               │
                               ▼
                    ┌─────────────────────┐
                    │    ANDROID PHONE    │
                    │                     │
                    │ BLE Receiver        │
                    │ Data Buffer         │
                    │ Feature Engineering │
                    │ Normalization       │
                    └──────────┬──────────┘
                               │
                         128 × Features
                               │
                               ▼
                    ┌─────────────────────┐
                    │   TensorFlow Lite   │
                    │                     │
                    │ CNN / LSTM /        │
                    │ Bi-LSTM / CNN-BiLSTM│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Activity Prediction │
                    └──────────┬──────────┘
                               │
                   ┌───────────┼───────────┐
                   ▼           ▼           ▼
                WALKING      FALL       RUNNING
                               │
                               ▼
                       Fall Confirmation
                               │
                    ┌──────────┼──────────┐
                    ▼          ▼          ▼
                   GPS        SMS       CLOUD
````

---

# 🔄 End-to-End Pipeline

```text
MPU6050
   ↓
Accelerometer + Gyroscope
   ↓
ESP32
   ↓
50 Hz Sampling
   ↓
Bluetooth Low Energy
   ↓
Android Application
   ↓
Data Buffer
   ↓
128-Sample Sliding Window
   ↓
Feature Engineering
   ↓
Normalization
   ↓
TensorFlow Lite
   ↓
Deep Learning Model
   ↓
Activity / Fall Prediction
   ↓
Fall Confirmation
   ↓
GPS + Emergency SMS
   ↓
Cloud Logging
```

---

# 🧠 Machine Learning Pipeline

```text
Dataset
   ↓
Data Loading
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Window Generation
   ↓
Train / Validation / Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
Best Model Selection
   ↓
TensorFlow Lite Conversion
   ↓
Android Deployment
```

---

# 📊 Dataset

The project can use inertial-sensor Human Activity Recognition datasets such as:

* **SisFall**
* **UCI Human Activity Recognition (UCI HAR)**

The dataset used for a particular experiment should be clearly documented along with its corresponding preprocessing and class-label definitions.

> The training and deployment pipelines must use consistent sensor features, sampling assumptions, window size and preprocessing.

---

# 📡 MPU6050

The MPU6050 provides six primary inertial sensor signals.

### Accelerometer

```text
AccX
AccY
AccZ
```

### Gyroscope

```text
GyroX
GyroY
GyroZ
```

These signals are continuously collected by the ESP32.

---

# ⚙️ Sampling Configuration

The proposed real-time configuration is:

```text
Sampling Frequency : 50 Hz
Window Size        : 128 samples
Window Duration    : 2.56 seconds
```

Calculation:

```text
128 / 50 = 2.56 seconds
```

The final deployment sampling frequency must match the assumptions used during model training.

---

# 🧮 Feature Engineering

The model input can contain **13 features**.

### Raw Features

```text
1. AccX
2. AccY
3. AccZ
4. GyroX
5. GyroY
6. GyroZ
```

### Engineered Features

```text
7.  AccMagnitude
8.  GyroMagnitude
9.  SMV
10. AccXY
11. AccYZ
12. AccXZ
13. GyroEnergy
```

Therefore:

```text
Input Shape = 128 × 13
```

---

# 📐 Feature Formulas

### Acceleration Magnitude

```text
AccMagnitude = √(AccX² + AccY² + AccZ²)
```

### Gyroscope Magnitude

```text
GyroMagnitude = √(GyroX² + GyroY² + GyroZ²)
```

### Signal Magnitude Vector

```text
SMV = √(AccMagnitude² + GyroMagnitude²)
```

### AccXY

```text
AccXY = √(AccX² + AccY²)
```

### AccYZ

```text
AccYZ = √(AccY² + AccZ²)
```

### AccXZ

```text
AccXZ = √(AccX² + AccZ²)
```

### Gyroscope Energy

```text
GyroEnergy = GyroX² + GyroY² + GyroZ²
```

---

# 🪟 Sliding Window

Continuous sensor data is divided into fixed-length windows.

```text
Sample 1
Sample 2
Sample 3
...
Sample 128
```

Each window becomes:

```text
128 × 13
```

For continuous inference, overlapping windows can be used.

Example:

```text
Window 1 → Samples 1–128
Window 2 → Samples 33–160
Window 3 → Samples 65–192
```

This allows continuous activity prediction.

---

# 🤖 Deep Learning Models

The project evaluates multiple deep-learning architectures:

```text
1. CNN
2. LSTM
3. CNN-LSTM
4. Bi-LSTM
5. CNN-BiLSTM
```

Models can be compared using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Cohen's Kappa
* MCC
* ROC-AUC
* Number of parameters
* Model size
* Inference latency
* Deployment feasibility

---

# 🧠 Lightweight CNN

A lightweight CNN can be used for resource-constrained deployment.

```text
Input
128 × 13
    ↓
Conv1D
32 Filters
    ↓
Batch Normalization
    ↓
Max Pooling
    ↓
Dropout
    ↓
Conv1D
64 Filters
    ↓
Batch Normalization
    ↓
Max Pooling
    ↓
Dropout
    ↓
Conv1D
128 Filters
    ↓
Batch Normalization
    ↓
Global Average Pooling
    ↓
Dense
128
    ↓
Dropout
    ↓
Output
```

---

# 🧬 CNN-BiLSTM

A larger CNN-BiLSTM architecture can be evaluated for mobile deployment.

```text
Input
128 × 13
    ↓
Conv1D
    ↓
Batch Normalization
    ↓
Max Pooling
    ↓
Conv1D
    ↓
Temporal Features
    ↓
Bi-LSTM
    ↓
Dense
    ↓
Dropout
    ↓
Output
```

The CNN extracts local motion patterns while the Bi-LSTM learns temporal dependencies.

---

# 📈 Model Evaluation

The models should be evaluated using:

```text
Accuracy
Precision
Recall
F1-Score
Cohen's Kappa
Matthews Correlation Coefficient
ROC-AUC
Confusion Matrix
```

### Model Comparison

| Model      | Accuracy | Precision | Recall | F1-Score | Parameters | Deployment |
| ---------- | -------: | --------: | -----: | -------: | ---------: | ---------- |
| CNN        |       -- |        -- |     -- |       -- |         -- | ✓          |
| LSTM       |       -- |        -- |     -- |       -- |         -- | ✓          |
| CNN-LSTM   |       -- |        -- |     -- |       -- |         -- | ✓          |
| Bi-LSTM    |       -- |        -- |     -- |       -- |         -- | ✓          |
| CNN-BiLSTM |       -- |        -- |     -- |       -- |         -- | ✓          |

> Replace the `--` values with experimentally measured results.

---

# 🔌 Hardware Components

## Required

```text
ESP32 DevKit
MPU6050
USB Data Cable
Jumper Wires
Breadboard
Android Smartphone
```

## Optional

```text
Li-ion / Li-Po Battery
TP4056 Charging Module
Buzzer
LED
Push Button
OLED Display
Wearable Enclosure
```

---

# 🔗 MPU6050 → ESP32 Wiring

Typical I²C wiring:

```text
MPU6050             ESP32
────────────────────────────
VCC                 3.3V
GND                 GND
SDA                 GPIO 21
SCL                 GPIO 22
```

```text
             MPU6050
          ┌───────────┐
          │           │
VCC ──────┤           ├──── 3.3V
GND ──────┤           ├──── GND
SDA ──────┤           ├──── GPIO 21
SCL ──────┤           ├──── GPIO 22
          └───────────┘
                 │
                 │ I²C
                 ▼
               ESP32
```

> Pin assignments may vary depending on the specific ESP32 board.

---

# 📟 ESP32 Responsibilities

The ESP32 performs:

```text
1. Initialize MPU6050
2. Read accelerometer
3. Read gyroscope
4. Maintain sampling rate
5. Create sensor packets
6. Establish BLE connection
7. Transmit sensor data
```

The Android smartphone performs the main deep-learning inference.

---

# 📶 BLE Communication

Communication:

```text
ESP32
  │
  │ Bluetooth Low Energy
  ▼
Android
```

Example packet:

```text
ax,ay,az,gx,gy,gz
```

Example:

```text
0.21,-0.84,9.72,0.02,-0.04,0.01
```

The Android application parses the packet and adds the sample to its sensor buffer.

---

# 📱 Android Application

The Android application handles:

```text
BLE Connection
      ↓
Sensor Data Reception
      ↓
Data Buffering
      ↓
Feature Engineering
      ↓
Normalization
      ↓
128-Sample Window
      ↓
TensorFlow Lite
      ↓
Prediction
      ↓
Fall Detection
      ↓
GPS
      ↓
Emergency Alert
```

---

# 📦 Android Project Structure

```text
android/
└── WearableAI/
    │
    ├── app/
    │
    └── src/
        └── main/
            ├── java/
            │   └── ...
            │
            ├── res/
            │   └── ...
            │
            └── assets/
                ├── model.tflite
                └── scaler_parameters.json
```

---

# 🪣 Android Data Buffer

The application continuously collects:

```text
Sample 1
Sample 2
Sample 3
...
Sample 128
```

Raw sensor data:

```text
128 × 6
```

After feature engineering:

```text
128 × 13
```

This becomes the model input.

---

# 📏 Normalization

The scaler fitted during training must be reused during mobile inference.

### Training

```text
Training Data
      ↓
Fit StandardScaler
      ↓
Save Mean + Scale
```

### Deployment

```text
Live Sensor Data
      ↓
Same Mean + Scale
      ↓
Normalized Data
```

Formula:

```text
x_scaled = (x - mean) / scale
```

The Android implementation must reproduce the training preprocessing exactly.

---

# 🧠 TensorFlow Lite

The selected Keras model is converted into TensorFlow Lite:

```text
Best Keras Model
       ↓
TensorFlow Lite
       ↓
model.tflite
```

Example:

```text
cnn_bilstm_best.keras
        ↓
model.tflite
```

The `.tflite` model is placed inside:

```text
android/app/src/main/assets/
```

---

# ⚡ Mobile Inference

The inference pipeline:

```text
128 × 13
     ↓
Normalization
     ↓
TensorFlow Lite Interpreter
     ↓
Deep Learning Model
     ↓
Output Probabilities
     ↓
Argmax
     ↓
Activity
```

Example output:

```text
Activity    : WALKING
Confidence  : XX.XX%
Status      : LIVE
```

> The displayed confidence must come from the actual model output.

---

# 🔁 Real-Time Sliding Window

Example overlapping windows:

```text
Window 1
1 ───────────────── 128

Window 2
33 ───────────────── 160

Window 3
65 ───────────────── 192
```

Each window generates a new prediction.

---

# 🚨 Fall Detection

A single uncertain prediction should not automatically trigger an emergency.

Proposed logic:

```text
Model Prediction
      ↓
Fall Class?
      ↓
Confidence Check
      ↓
Consecutive Predictions
      ↓
Potential Fall
      ↓
User Confirmation
      ↓
Emergency Alert
```

Example interface:

```text
┌─────────────────────────────┐
│      ⚠ FALL DETECTED        │
│                             │
│ Are you okay?               │
│                             │
│ Alert in 10 seconds         │
│                             │
│          I'M OK             │
└─────────────────────────────┘
```

If the user confirms they are safe, the emergency alert can be cancelled.

---

# 📍 GPS Location

After a confirmed fall:

```text
Android
   ↓
Location Services
   ↓
Latitude + Longitude
   ↓
Emergency Location
```

The application can provide the location to an emergency contact.

---

# 📩 Emergency SMS

Example:

```text
⚠ EMERGENCY ALERT

A possible fall has been detected.

Please check on the user.

Current location:
[Map Location]
```

The exact implementation depends on Android permissions and the chosen communication method.

---

# ☁️ Cloud Logging

Optional cloud integration can store:

```text
Timestamp
Activity
Confidence
Fall Event
Location
```

Example:

```text
Timestamp  : 2026-09-16 11:30
Activity   : WALKING
Confidence : XX.XX%
Fall       : NO
```

---

# 📊 Android Dashboard

Proposed real-time interface:

```text
╔══════════════════════════════════╗
║       WEARABLE AI MONITOR        ║
╠══════════════════════════════════╣
║                                  ║
║       ● DEVICE CONNECTED         ║
║                                  ║
║       CURRENT ACTIVITY           ║
║                                  ║
║            WALKING               ║
║                                  ║
║       Confidence: XX.XX%         ║
║                                  ║
║       Samples: 128               ║
║       BLE: Connected             ║
║       Sensor: MPU6050            ║
║       Model: CNN-BiLSTM          ║
║                                  ║
║       GPS: Ready                 ║
║       Emergency: Ready           ║
║                                  ║
╚══════════════════════════════════╝
```

---

# 📁 Repository Structure

```text
Wearable-AI-Fall-Detection/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── keras/
│   └── tflite/
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── windowing.py
│   ├── feature_engineering.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── convert_tflite.py
│
├── esp32/
│   ├── wearable_sensor/
│   │   └── wearable_sensor.ino
│   └── README.md
│
├── android/
│   └── WearableAI/
│
├── notebooks/
│   ├── exploration.ipynb
│   └── experiments.ipynb
│
├── results/
│   ├── confusion_matrix/
│   ├── metrics/
│   └── training_curves/
│
├── hardware/
│   ├── circuit_diagram.png
│   ├── wiring.png
│   └── prototype.jpg
│
└── docs/
    ├── architecture.png
    ├── methodology.md
    └── deployment.md
```

---

# 🛠️ Technologies Used

### Artificial Intelligence

```text
Python
TensorFlow
Keras
NumPy
Pandas
Scikit-learn
Matplotlib
```

### Embedded Systems

```text
ESP32
MPU6050
Arduino IDE
I²C
Bluetooth Low Energy
```

### Mobile

```text
Android Studio
Kotlin
TensorFlow Lite
Android BLE APIs
Android Location APIs
```

### Cloud

```text
Firebase / Cloud Backend
```

---

# 💻 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Wearable-AI-Fall-Detection.git
```

Enter the project:

```bash
cd Wearable-AI-Fall-Detection
```

Create virtual environment:

```bash
python -m venv .venv
```

Activate on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Machine Learning Commands

### Data Loading

```bash
python src/data_loader.py
```

### Preprocessing

```bash
python src/preprocessing.py
```

### Window Generation

```bash
python src/windowing.py
```

### Training

```bash
python src/train.py
```

### Evaluation

```bash
python src/evaluate.py
```

### TensorFlow Lite Conversion

```bash
python src/convert_tflite.py
```

---

# 🔬 Complete ML Experiment Pipeline

```text
Dataset
   ↓
Data Loader
   ↓
Preprocessing
   ↓
Feature Engineering
   ↓
Windowing
   ↓
Train/Test Split
   ↓
CNN
   ↓
LSTM
   ↓
CNN-LSTM
   ↓
Bi-LSTM
   ↓
CNN-BiLSTM
   ↓
Performance Comparison
   ↓
Best Model
   ↓
TFLite Conversion
   ↓
Android Deployment
```

---

# 🔧 Complete Hardware Pipeline

```text
MPU6050
   ↓
ESP32
   ↓
50 Hz Sampling
   ↓
BLE
   ↓
Android
   ↓
128-Sample Buffer
   ↓
Feature Engineering
   ↓
Normalization
   ↓
TensorFlow Lite
   ↓
CNN / CNN-BiLSTM
   ↓
Activity Prediction
```

---

# 🚨 Complete Fall-Alert Pipeline

```text
MPU6050
   ↓
ESP32
   ↓
BLE
   ↓
Android
   ↓
128 × 13
   ↓
TensorFlow Lite
   ↓
Fall Prediction
   ↓
Confidence Check
   ↓
Consecutive Prediction Check
   ↓
User Confirmation
   ↓
GPS
   ↓
Emergency SMS
   ↓
Cloud Logging
```

---

# 🧪 Testing

The system should be tested using controlled and safe experiments.

### Activity Testing

```text
Standing
Walking
Sitting
Running
Other Dataset Activities
```

### Fall Testing

Fall detection experiments must be performed safely using controlled conditions.

Never intentionally perform dangerous falls for testing.

---

# 📈 Deployment Metrics

The final project should report experimentally measured values for:

```text
Classification Accuracy
Precision
Recall
F1 Score
Confusion Matrix
Model Parameters
Model Size
TFLite Size
Inference Latency
BLE Latency
End-to-End Latency
Battery Runtime
```

Example:

```text
Model Accuracy     : XX.XX%
F1 Score           : XX.XX%
Model Size         : XX MB
Inference Latency  : XX ms
BLE Latency        : XX ms
```

---

# 🗺️ Development Roadmap

## Phase 1 — Machine Learning

* [x] Dataset loader
* [x] Data preprocessing
* [x] Feature engineering
* [x] Sliding-window generation
* [ ] Train candidate models
* [ ] Compare models
* [ ] Select final model
* [ ] Final evaluation

---

## Phase 2 — TensorFlow Lite

* [ ] Convert selected Keras model
* [ ] Verify TFLite output
* [ ] Compare Keras vs TFLite predictions
* [ ] Measure model size
* [ ] Measure inference latency
* [ ] Optimize model if required

---

## Phase 3 — Hardware

* [ ] Assemble ESP32 + MPU6050
* [ ] Verify I²C communication
* [ ] Read accelerometer
* [ ] Read gyroscope
* [ ] Configure 50 Hz sampling
* [ ] Generate sensor packets

---

## Phase 4 — BLE

* [ ] Implement ESP32 BLE peripheral
* [ ] Create BLE service
* [ ] Create sensor characteristic
* [ ] Test BLE connection
* [ ] Stream sensor data
* [ ] Verify packet integrity

---

## Phase 5 — Android

* [ ] Create Android project
* [ ] Implement BLE scanning
* [ ] Connect ESP32
* [ ] Receive sensor data
* [ ] Create sensor buffer
* [ ] Implement 128-sample window
* [ ] Implement feature engineering
* [ ] Implement normalization
* [ ] Integrate TensorFlow Lite

---

## Phase 6 — Real-Time AI

* [ ] Run continuous inference
* [ ] Display activity
* [ ] Display confidence
* [ ] Implement sliding windows
* [ ] Measure inference latency
* [ ] Test real sensor data
* [ ] Compare live vs dataset performance

---

## Phase 7 — Emergency System

* [ ] Fall confidence threshold
* [ ] Consecutive prediction logic
* [ ] User confirmation screen
* [ ] GPS integration
* [ ] Emergency SMS
* [ ] Emergency contacts
* [ ] Cloud logging

---

## Phase 8 — Final Wearable

* [ ] Battery power
* [ ] Compact wiring
* [ ] Wearable enclosure
* [ ] Wrist / waist mounting
* [ ] Android dashboard
* [ ] End-to-end testing
* [ ] Final demonstration
* [ ] Documentation
* [ ] Project presentation

---

# 🔐 Privacy & Safety

The system may process motion and location information.

Recommended practices:

* Obtain user consent before collecting data.
* Minimize personal data collection.
* Protect cloud-stored information.
* Never publish private location information.
* Never commit API keys or credentials.
* Keep emergency contact information private.
* Do not upload private sensor recordings.
* Treat the system as an academic/research prototype.
* Do not represent the system as a certified medical device.
* Perform physical fall experiments safely.

---

# 🚫 Files Not to Upload to GitHub

```text
.venv/
venv/
__pycache__/
*.pyc
.pytest_cache/

data/raw/
data/SisFall_dataset/
large datasets

*.keras
*.h5

.env
google-services.json

API keys
Passwords
Private GPS data
Private sensor recordings
```

---

# 📝 .gitignore

```gitignore
# Python
.venv/
venv/
__pycache__/
*.pyc
.pytest_cache/

# Jupyter
.ipynb_checkpoints/

# Data
data/raw/
data/SisFall_dataset/
data/processed/*.csv
data/processed/*.npy

# Large Models
models/*.keras
models/*.h5

# Logs
*.log
*.tmp

# IDE
.vscode/
.idea/

# Android
android/.gradle/
android/build/
android/app/build/

# Secrets
.env
google-services.json

# Temporary files
tmp_*
```

---

# 🌟 Future Improvements

* Transformer-based time-series models
* Knowledge distillation
* Model quantization
* Personalized activity recognition
* Adaptive fall detection thresholds
* TinyML deployment
* Battery optimization
* Smartwatch integration
* Additional wearable sensors
* Cloud analytics dashboard
* Personalized models
* Federated learning
* Edge-cloud hybrid inference

---

# 💡 Project Vision

```text
              SENSE
                ↓
             CONNECT
                ↓
            UNDERSTAND
                ↓
             DETECT
                ↓
             PROTECT
```

The long-term vision is to build a wearable AI system that transforms real-time inertial sensor signals into intelligent activity understanding and timely emergency assistance.

---

# 👨‍💻 Author

## Pranesh M

**B.Tech Artificial Intelligence**

**Amrita Vishwa Vidyapeetham**

Coimbatore, India

---

# 📄 License

This project is intended for academic and research purposes.

Choose and add an appropriate open-source license before public distribution.

---

<p align="center">

<b>🚀 Wearable AI — From Sensor Signals to Real-Time Intelligence</b>

</p>
```
