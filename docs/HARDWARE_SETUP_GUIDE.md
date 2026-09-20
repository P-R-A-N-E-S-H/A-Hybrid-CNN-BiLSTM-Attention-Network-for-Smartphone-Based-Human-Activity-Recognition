# Hardware Setup, Wiring & Calibration Guide

**System:** ESP32 / Arduino Uno + MPU-6050 6-DOF IMU Sensor Telemetry  
**Sampling Rate:** 50 Hz (20 ms period)  
**Baud Rate:** 115200 bps  

---

## 1. Hardware Bill of Materials (BOM)

| Item | Component | Specification | Purpose |
| :--- | :--- | :--- | :--- |
| **1** | **Microcontroller** | ESP32 NodeMCU / ESP32-WROOM-32 (or Arduino Uno/Nano) | Telemetry acquisition & streaming |
| **2** | **IMU Sensor** | InvenSense MPU-6050 Module (GY-521) | 3-Axis Accel ($\pm 2g$) + 3-Axis Gyro ($\pm 250^\circ/s$) |
| **3** | **Interconnects** | 4-Pin Female-to-Female Jumper Wires | I2C & Power connections |
| **4** | **Cable** | Micro-USB to USB-A data cable | High-speed serial telemetry to PC |
| **5** | **Mounting Strap** | Elastic wearable strap / wristband | Secure sensor attachment to waist/wrist |

---

## 2. Wiring & Pinout Diagram

### 2.1 ESP32 Connection Table

```
   ESP32 NodeMCU                 MPU-6050 (GY-521)
┌──────────────────┐           ┌──────────────────┐
│             3.3V ├───────────┤ VCC              │
│              GND ├───────────┤ GND              │
│          GPIO 21 ├───────────┤ SDA (Data)       │
│          GPIO 22 ├───────────┤ SCL (Clock)      │
│          GPIO 19 ├──(Opt)────┤ INT (Interrupt)  │
└──────────────────┘           └──────────────────┘
```

| ESP32 Pin | MPU-6050 Pin | Wire Color (Typical) | Function |
| :--- | :--- | :--- | :--- |
| **3V3** | **VCC** | Red | 3.3V Regulated Power |
| **GND** | **GND** | Black | System Ground |
| **GPIO 21** | **SDA** | Green / Blue | I2C Serial Data line |
| **GPIO 22** | **SCL** | Yellow / White | I2C Serial Clock line |
| **GPIO 2 (LED)** | Internal | — | Status heartbeat blink (1Hz) |

### 2.2 Arduino Uno / Nano Connection Table

| Arduino Pin | MPU-6050 Pin | Function |
| :--- | :--- | :--- |
| **5V / 3.3V** | **VCC** | Power Supply |
| **GND** | **GND** | Ground |
| **A4** | **SDA** | I2C Data |
| **A5** | **SCL** | I2C Clock |

---

## 3. Firmware Flashing Instructions

1. **Open Arduino IDE** (or VS Code + PlatformIO).
2. Install the **ESP32 Board Package** in Board Manager if using ESP32.
3. Open `hardware/firmware/esp32_har_streamer/esp32_har_streamer.ino`.
4. Select Port (e.g. `COM3` on Windows) and Board: `ESP32 Dev Module`.
5. Click **Upload**.
6. When upload finishes, open the Serial Monitor at **115200 baud**.
7. Keep the board stationary on a flat table during the 2-second automatic zero-bias calibration sequence.

---

## 4. Telemetry Stream Protocol

The firmware outputs comma-separated values (CSV) at 50 Hz:
```text
Ax,Ay,Az,Gx,Gy,Gz,Timestamp_ms\n
```
Example raw packet:
```text
0.01245,0.98120,0.03412,0.00120,-0.00340,0.00085,124850
```

- **Ax, Ay, Az:** Accelerometer values in units of Earth gravity ($g \approx 9.81\,\text{m/s}^2$).
- **Gx, Gy, Gz:** Gyroscope angular velocity values in radians per second ($\text{rad/s}$).
- **Timestamp_ms:** Monotonic microcontroller millisecond counter.

---

## 5. Live Review Demonstration Workflow

1. Plug in the ESP32 USB cable.
2. Launch the backend API & Web Server:
   ```bash
   python api/server.py
   ```
3. Open your browser at `http://localhost:8000`.
4. In the **Sensor Source** deck, select **Physical COM**, choose your COM port, and click **Connect Port**.
5. Move the sensor (walk, stand, sit, shake) and observe the live multi-channel waveforms and real-time classification updates!
