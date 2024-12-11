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

## **Installation**
- **Step 1**: Ensure that you install app.py, the static folder, and the templats folder into one folder.
- **Step 2**: Run app.py in any IDE (VSCode, Pycharm, etc.)
- **Step 3**: Once you run it, you will see a URL pop open. Ensure you can click it by using CTRL + Click (PC) or Command + Click (Mac)

---

## **Dataset**
- The data generated consists of a CSV of 32 landmarks that corresponds to different points on the body
- ![image](https://github.com/user-attachments/assets/494efd57-d1cd-4400-88e9-01198bb2ccea)

---

## **Output Examples**
- **Camera Feed**: It should output the number of repetitions (reps) for a workout, angle measurements, and display feedback for every frame
- **Post-Workout Feedback**: A summary on the number of reps, % of form accuracy for the entire workout, and most common feedback

---


