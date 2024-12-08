# **Computer Vision Workout Tracker**

![Workout Tracker Demo](demo.gif)

A real-time workout tracking application using computer vision to analyze shoulder presses. This tool provides immediate feedback on form, tracks repetitions, and records workout data for further analysis. Built with **MediaPipe**, **OpenCV**, and **Python**, this project is a powerful tool for fitness enthusiasts and trainers.

---

## **Table of Contents**
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Output Examples](#output-examples)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## **Overview**
The **Computer Vision Workout Tracker** leverages pose estimation and video processing to track key body movements during shoulder presses. This project provides:
- Real-time visual feedback with color-coded exercise states.
- Accurate repetition counting.
- A CSV file containing detailed workout data for performance analysis.

---

## **Features**
- **Real-Time Tracking**: Detects key body landmarks such as shoulders, elbows, and wrists.
- **State Classification**:
  - **RESTING**: Wrist is below the midpoint between the shoulder and hip.
  - **FLEXED**: Arm is bent during the press motion.
  - **EXTENDED**: Arm is straight, typically at the top of the press.
- **Color-Coded Feedback**:
  - **Green**: RESTING
  - **Yellow**: FLEXED
  - **Red**: EXTENDED
- **Live Camera Feed Integration**: Tracks movements in real-time using a webcam.
- **Video Analysis**: Analyze pre-recorded workout videos.
- **Data Logging**: Saves workout data (keypoints, angles, reps, states) to a CSV file.

---
