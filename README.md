# Virtual Vahana: Integrated Decision-Intelligence ADAS

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?style=for-the-badge&logo=pytorch)
![CARLA](https://img.shields.io/badge/CARLA-Simulator-1d2021?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Phase_1_Complete-success?style=for-the-badge)

**Event:** Virtual Vahana Contest 2026 (Phase 1 Technical Submission)  
**Institution:** Amrita Vishwa Vidyapeetham, Kollam, Kerala  
**Team Members:** Y Sai Sailesh Reddy | Bhavana PH | Sidharth R Krishna  

---

## 🚙 Project Overview
This repository contains a closed-loop Advanced Driver Assistance System (ADAS) engineered for the CARLA autonomous driving simulator. The system features a custom, GPU-accelerated perception pipeline that fuses **YOLOv8** (Dynamic Object Detection) and **UFLDv2** (Ultra Fast Lane Detection). 

To combat simulator domain shift and screen-space latency, this project discards traditional temporal smoothing (EMA/Kalman filters) in favor of a **Zero-Latency Perspective Mapping Algorithm**. By mathematically translating 2D row-classifications into 3D simulator space using custom Field-of-View (FOV) scalars, the system guarantees physical lane adherence and drives a highly responsive arbitration engine.

### 🌟 Key Technical Features
* **Zero-Latency Lane Tracking:** Instantaneous polynomial curve fitting mapping 2D UFLD classifications directly to the 3D CARLA environment.
* **Domain Shift Calibration:** Custom FOV scaling (1.25x) and linear horizon mapping to adapt real-world 60° dashcam training weights to 90° simulated lenses.
* **Bulletproof LDW:** Bumper-level lane offset calculation utilizing a dynamic single-line tracking fallback to prevent line-of-sight failures during high-frequency maneuvers.
* **Time-to-Collision (TTC) Engine:** RGB and Depth sensor fusion generating real-time closing velocities to actuate Autonomous Emergency Braking (AEB).

## ⚙️ System Architecture
The system is divided into three asynchronous layers:
1. **Perception Layer (RTX 5070 Ti):** Ingests raw sensor data, strips the sky via extrinsic cropping, and evaluates the tensors through ResNet-18 (UFLDv2) and YOLOv8 simultaneously.
2. **Feature Logic Layer (CPU):** Evaluates cross-track error and TTC based on strictly spatial data coordinates.
3. **Arbitration Matrix:** A deterministic state machine ensuring longitudinal safety (FCW/AEB) overrides lateral precision (LKA/LDW).

## 🛠️ Prerequisites & Installation

**1. Clone the Repository**
```bash
git clone https://github.com/sailesh2408/VirtualVahana.git
cd VirtualVahana
```

**2. Setup the Virtual Environment**
Ensure you are using Python 3.10+ (Developed and tested on Python 3.12).
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```
*Note: Ensure your CARLA PythonAPI `.egg` file is properly exported to your `PYTHONPATH` or placed in your site-packages for the simulator connection to work.*

## 🚀 Usage Guide

**1. Start the CARLA Server**
Navigate to your CARLA installation directory and launch the server in synchronous mode:
```bash
./CarlaUE4.sh -quality-level=Epic
```

**2. Run the ADAS Pipeline**
In your project workspace, execute the main control script:
```bash
python main.py
```

**Controls inside the PyGame Window:**
* The vehicle will begin autonomous tracking automatically.
* The HUD overlay will display real-time Lane Status (Secure/Warning), TTC Alerts, and ADAS Trust percentages.

## 📄 License & Acknowledgments
* **UFLDv2 Architecture:** Inspired by Qin et al.'s work on structure-aware deep lane detection.
* **Object Detection:** Ultralytics YOLOv8.
* Developed for the Virtual Vahana autonomous driving challenge.
