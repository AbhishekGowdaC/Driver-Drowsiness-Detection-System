# 🚗 Driver Drowsiness Detection System

An AI-powered Driver Drowsiness Detection System that monitors the driver's facial landmarks in real time using Computer Vision.

## 📌 Overview

This project detects signs of driver fatigue using a webcam and facial landmark detection.

The system continuously monitors:

- 👁️ Eye closure (EAR)
- 😮 Yawning (MAR)
- 😊 Facial landmarks
- 🧠 Head pose estimation
- 😴 PERCLOS
- 👀 Blink rate
- 🎮 Steering activity (simulation)

When drowsiness is detected, the system:

- 🔊 Plays an alarm
- 🗣️ Gives voice alerts
- 🚨 Activates hazard mode
- 📉 Simulates vehicle speed reduction
- 📝 Logs events into a CSV file

---

## 🚀 Features

- Real-time Face Detection
- Eye Aspect Ratio (EAR)
- Mouth Aspect Ratio (MAR)
- Head Pose Estimation
- Blink Detection
- PERCLOS Calculation
- Voice Alerts
- Alarm System
- CSV Logging

---

## 🛠️ Technologies Used

- Python
- OpenCV
- Dlib
- NumPy
- Pygame
- pyttsx3

---

## 📂 Project Structure

```text
Driver-Drowsiness-Detection-System/
│
├── main.py
├── README.md
├── requirements.txt
├── alarm.mp3
├── tick.mp3
├── drowsiness_log.csv
├── shape_predictor_68_face_landmarks.dat
└── .gitignore
```

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/AbhishekGowdaC/Driver-Drowsiness-Detection-System.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run

```bash
python main.py
```

---

## 📸 Screenshots

### Driver Monitoring System

(Add screenshot here)

### Face Detection

(Add screenshot here)

### Drowsiness Alert

(Add screenshot here)

---

## ⚠️ Model File

This project uses the Dlib facial landmark model:

`shape_predictor_68_face_landmarks.dat`
The model file is approximately **95 MB**. Because of its size, it is recommended **not** to store it directly in the GitHub repository.

Instead:

1. Download `shape_predictor_68_face_landmarks.dat` from the official Dlib model repository.
2. Place the downloaded file inside the project folder (or a `models/` folder).
3. Update the file path in `main.py` if you move it into a different directory.

---

## 👨‍💻 Author

**Abhishek Gowda C**

Artificial Intelligence & Machine Learning Engineer
