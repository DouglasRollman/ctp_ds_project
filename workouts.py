import cv2
import mediapipe as mp
import math
from collections import deque
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt

# Initialize mediapipe Pose module
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    ab = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]
    dot_product = ab[0] * bc[0] + ab[1] * bc[1]
    mag_ab = math.sqrt(ab[0]**2 + ab[1]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)
    cos_angle = dot_product / (mag_ab * mag_bc)
    angle = math.acos(cos_angle) * 180.0 / math.pi
    return angle

# Function to smooth angles using a moving average
def smooth_angle(angle, window):
    window.append(angle)
    return np.mean(window)

def track_shoulder_presses(video_source=0, output_csv="live_keypoints_data.csv"):
    import cv2
    import csv
    from collections import deque
    import mediapipe as mp

    # Open the CSV file for writing
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(["Frame", "Time", "Left_Shoulder_X", "Left_Shoulder_Y", "Left_Elbow_X", "Left_Elbow_Y",
                         "Left_Wrist_X", "Left_Wrist_Y", "Right_Shoulder_X", "Right_Shoulder_Y",
                         "Right_Elbow_X", "Right_Elbow_Y", "Right_Wrist_X", "Right_Wrist_Y",
                         "Left_Angle", "Right_Angle", "Left_Wrist_Y", "Right_Wrist_Y", "Left_State", "Right_State"])

        # Video feed
        cap = cv2.VideoCapture(video_source)

        # Get the frame rate of the live feed (if available)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30  # Default to 30 if unavailable

        # Mediapipe instance
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            # Rep counters and states
            left_rep_count, right_rep_count = 0, 0
            left_state, right_state = "RESTING", "RESTING"
            left_angle_window, right_angle_window = deque(maxlen=5), deque(maxlen=5)

            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame.")
                    break

                # Calculate the elapsed time
                elapsed_time = frame_count / fps

                # Recolor image
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)

                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Extract landmarks
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                    # Get coordinates for left arm
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1],
                                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0]]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * frame.shape[1],
                                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * frame.shape[0]]
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * frame.shape[1],
                                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * frame.shape[0]]
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * frame.shape[1],
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * frame.shape[0]]

                    # Get coordinates for right arm
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * frame.shape[1],
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * frame.shape[0]]
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * frame.shape[1],
                                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * frame.shape[0]]
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * frame.shape[1],
                                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * frame.shape[0]]

                    # Calculate midpoints for "RESTING" logic
                    left_midpoint = (left_shoulder[1] + left_hip[1]) / 2
                    right_midpoint = (right_shoulder[1] + right_hip[1]) / 2

                    # Calculate and smooth angles
                    left_angle = smooth_angle(calculate_angle(left_shoulder, left_elbow, left_wrist), left_angle_window)
                    right_angle = smooth_angle(calculate_angle(right_shoulder, right_elbow, right_wrist), right_angle_window)

                    # State logic for left arm
                    if left_wrist[1] > left_midpoint:  # Wrist below midpoint
                        left_state = "RESTING"
                    elif left_state == "RESTING" and left_wrist[1] <= left_midpoint:  # Wrist moves above midpoint
                        left_state = "EXTENDED"
                    elif left_state == "EXTENDED" and left_angle < 100:
                        left_state = "FLEXED"
                    elif left_state == "FLEXED" and left_angle > 160:
                        left_state = "EXTENDED"
                        left_rep_count += 1

                    # State logic for right arm
                    if right_wrist[1] > right_midpoint:  # Wrist below midpoint
                        right_state = "RESTING"
                    elif right_state == "RESTING" and right_wrist[1] <= right_midpoint:  # Wrist moves above midpoint
                        right_state = "EXTENDED"
                    elif right_state == "EXTENDED" and right_angle < 100:
                        right_state = "FLEXED"
                    elif right_state == "FLEXED" and right_angle > 160:
                        right_state = "EXTENDED"
                        right_rep_count += 1

                    # Write data to CSV (each frame)
                    writer.writerow([frame_count, elapsed_time, left_shoulder[0], left_shoulder[1], left_elbow[0],
                                     left_elbow[1], left_wrist[0], left_wrist[1], right_shoulder[0], right_shoulder[1],
                                     right_elbow[0], right_elbow[1], right_wrist[0], right_wrist[1],
                                     left_angle, right_angle, left_wrist[1], right_wrist[1], left_state, right_state])

                    frame_count += 1

                # Render detections with color-coded skeletons
                if results.pose_landmarks:
                    left_color = (0, 255, 0) if left_state == "RESTING" else (0, 255, 255) if left_state == "FLEXED" else (0, 0, 255)
                    right_color = (0, 255, 0) if right_state == "RESTING" else (0, 255, 255) if right_state == "FLEXED" else (0, 0, 255)

                    # Draw landmarks with state-based colors
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=left_color, thickness=2, circle_radius=2),
                                              mp_drawing.DrawingSpec(color=right_color, thickness=2, circle_radius=2))

                # Display rep counters and current states on the frame
                cv2.putText(image, f"Left Reps: {left_rep_count} ({left_state})", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image, f"Right Reps: {right_rep_count} ({right_state})", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Display the frame
                cv2.imshow('Live Shoulder Press Tracker', image)

                if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'Q' to exit
                    break

            cap.release()
            cv2.destroyAllWindows()


# Initialize mediapipe Pose module
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    ab = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]
    dot_product = ab[0] * bc[0] + ab[1] * bc[1]
    mag_ab = math.sqrt(ab[0]**2 + ab[1]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)
    cos_angle = dot_product / (mag_ab * mag_bc)
    angle = math.acos(cos_angle) * 180.0 / math.pi
    return angle

# Function to smooth angles using a moving average
def smooth_angle(angle, window):
    window.append(angle)
    return np.mean(window)

import cv2
import mediapipe as mp
import math
import numpy as np
from collections import deque
import csv

# Initialize mediapipe Pose module
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    ab = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]
    dot_product = ab[0] * bc[0] + ab[1] * bc[1]
    mag_ab = math.sqrt(ab[0]**2 + ab[1]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)
    cos_angle = dot_product / (mag_ab * mag_bc)
    angle = math.acos(cos_angle) * 180.0 / math.pi
    return angle

# Function to smooth angles using a moving average
def smooth_angle(angle, window):
    window.append(angle)
    return np.mean(window)

def track_shoulder_presses_on_video(input_video, output_video, csv_output):
    import cv2
    import csv
    from collections import deque
    import mediapipe as mp

    # Open the input video
    cap = cv2.VideoCapture(input_video)

    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video information
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize VideoWriter to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    # Initialize CSV file to save keypoints and angles
    with open(csv_output, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(["Frame", "Time", "Left_Shoulder_X", "Left_Shoulder_Y", "Left_Elbow_X", "Left_Elbow_Y",
                         "Left_Wrist_X", "Left_Wrist_Y", "Right_Shoulder_X", "Right_Shoulder_Y", "Right_Elbow_X",
                         "Right_Elbow_Y", "Right_Wrist_X", "Right_Wrist_Y", "Left_Angle", "Right_Angle",
                         "Left_Wrist_Y", "Right_Wrist_Y", "Left_State", "Right_State"])

        # Mediapipe instance
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            # Rep counters and states
            left_rep_count, right_rep_count = 0, 0
            left_state, right_state = "RESTING", "RESTING"
            left_angle_window, right_angle_window = deque(maxlen=5), deque(maxlen=5)

            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame.")
                    break

                # Calculate the elapsed time
                elapsed_time = frame_count / fps

                # Recolor image
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)

                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Extract landmarks
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                    # Get coordinates for left arm
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1],
                                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0]]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * frame.shape[1],
                                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * frame.shape[0]]
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * frame.shape[1],
                                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * frame.shape[0]]
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * frame.shape[1],
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * frame.shape[0]]

                    # Get coordinates for right arm
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * frame.shape[1],
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * frame.shape[0]]
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * frame.shape[1],
                                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * frame.shape[0]]
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * frame.shape[1],
                                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * frame.shape[0]]

                    # Calculate midpoints for "RESTING" logic
                    left_midpoint = (left_shoulder[1] + left_hip[1]) / 2
                    right_midpoint = (right_shoulder[1] + right_hip[1]) / 2

                    # Calculate and smooth angles
                    left_angle = smooth_angle(calculate_angle(left_shoulder, left_elbow, left_wrist), left_angle_window)
                    right_angle = smooth_angle(calculate_angle(right_shoulder, right_elbow, right_wrist), right_angle_window)

                    # State logic for left arm
                    if left_wrist[1] > left_midpoint:  # Wrist below midpoint
                        left_state = "RESTING"
                    elif left_state == "RESTING" and left_wrist[1] <= left_midpoint:  # Wrist moves above midpoint
                        left_state = "EXTENDED"
                    elif left_state == "EXTENDED" and left_angle < 100:
                        left_state = "FLEXED"
                    elif left_state == "FLEXED" and left_angle > 160:
                        left_state = "EXTENDED"
                        left_rep_count += 1

                    # State logic for right arm
                    if right_wrist[1] > right_midpoint:  # Wrist below midpoint
                        right_state = "RESTING"
                    elif right_state == "RESTING" and right_wrist[1] <= right_midpoint:  # Wrist moves above midpoint
                        right_state = "EXTENDED"
                    elif right_state == "EXTENDED" and right_angle < 100:
                        right_state = "FLEXED"
                    elif right_state == "FLEXED" and right_angle > 160:
                        right_state = "EXTENDED"
                        right_rep_count += 1

                    # Write data to CSV (each frame)
                    writer.writerow([frame_count, elapsed_time, left_shoulder[0], left_shoulder[1], left_elbow[0],
                                     left_elbow[1], left_wrist[0], left_wrist[1], right_shoulder[0], right_shoulder[1], 
                                     right_elbow[0], right_elbow[1], right_wrist[0], right_wrist[1],
                                     left_angle, right_angle, left_wrist[1], right_wrist[1], left_state, right_state])

                    frame_count += 1

                # Render detections with color-coded skeletons
                if results.pose_landmarks:
                    # Assign colors based on states
                    left_color = (0, 255, 0) if left_state == "RESTING" else (0, 255, 255) if left_state == "FLEXED" else (0, 0, 255)
                    right_color = (0, 255, 0) if right_state == "RESTING" else (0, 255, 255) if right_state == "FLEXED" else (0, 0, 255)

                    # Draw landmarks with state-based colors
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=left_color, thickness=2, circle_radius=2),
                                              mp_drawing.DrawingSpec(color=right_color, thickness=2, circle_radius=2))

                # Display rep counters and current states on the frame
                cv2.putText(image, f"Left Reps: {left_rep_count} ({left_state})", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image, f"Right Reps: {right_rep_count} ({right_state})", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Write the processed frame to the output video
                out.write(image)

                # Display the frame
                cv2.imshow('Shoulder Press Tracker', image)

                if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'Q' to exit
                    break