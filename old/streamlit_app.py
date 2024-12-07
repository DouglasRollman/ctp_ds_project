import pandas as pd
import streamlit as st
import cv2
import csv
import tempfile
import mediapipe as mp
import numpy as np
import time
from collections import deque
import matplotlib.pyplot as plt


# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    ab = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]
    dot_product = ab[0] * bc[0] + ab[1] * bc[1]
    mag_ab = np.sqrt(ab[0]**2 + ab[1]**2)
    mag_bc = np.sqrt(bc[0]**2 + bc[1]**2)
    cos_angle = dot_product / (mag_ab * mag_bc)
    angle = np.arccos(cos_angle) * 180.0 / np.pi
    return angle


# Function to smooth angles using a moving average
def smooth_angle(angle, window):
    window.append(angle)
    return np.mean(window)


# Shoulder Press Tracker function integrated with Streamlit
def track_shoulder_presses_on_video(input_video, output_video, csv_output):

    # Open the input video
    cap = cv2.VideoCapture(input_video)

    # Check if the video was opened successfully
    if not cap.isOpened():
        st.error("Error: Could not open video.")
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
        writer.writerow(["Frame", "Time", "Left_Shoulder_X", "Left_Shoulder_Y", "Left_Elbow_X", "Left_Elbow_Y",
                         "Left_Wrist_X", "Left_Wrist_Y", "Right_Shoulder_X", "Right_Shoulder_Y", "Right_Elbow_X",
                         "Right_Elbow_Y", "Right_Wrist_X", "Right_Wrist_Y", "Left_Angle", "Right_Angle",
                         "Left_State", "Right_State"])

        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            left_rep_count, right_rep_count = 0, 0
            left_state, right_state = "RESTING", "RESTING"
            left_angle_window, right_angle_window = deque(maxlen=5), deque(maxlen=5)

            frame_count = 0

            # Create a placeholder for video display in Streamlit
            frame_placeholder = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to grab frame.")
                    break

                elapsed_time = frame_count / fps

                # Process frame
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                    # Get left and right arm coordinates
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1],
                                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0]]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * frame.shape[1],
                                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * frame.shape[0]]
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * frame.shape[1],
                                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * frame.shape[0]]
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * frame.shape[1],
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * frame.shape[0]]

                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * frame.shape[1],
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * frame.shape[0]]
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * frame.shape[1],
                                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * frame.shape[0]]
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * frame.shape[1],
                                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * frame.shape[0]]

                    # Calculate angles and states
                    left_angle = smooth_angle(calculate_angle(left_shoulder, left_elbow, left_wrist), left_angle_window)
                    right_angle = smooth_angle(calculate_angle(right_shoulder, right_elbow, right_wrist), right_angle_window)

                    # Update rep counts and states
                    left_state, left_rep_count = update_state(left_state, left_angle, left_wrist, left_hip, left_rep_count, left_shoulder)
                    right_state, right_rep_count = update_state(right_state, right_angle, right_wrist, right_hip, right_rep_count, right_shoulder)

                    # Write data to CSV
                    writer.writerow([frame_count, elapsed_time, *left_shoulder, *left_elbow, *left_wrist,
                                     *right_shoulder, *right_elbow, *right_wrist, left_angle, right_angle,
                                     left_state, right_state])

                frame_count += 1
                out.write(image)

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

                # Display the frame in Streamlit
                frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

            cap.release()
            out.release()


# Helper function to update state and rep count
def update_state(state, angle, wrist, hip, rep_count, shoulder):
    midpoint = (shoulder[1] + hip[1]) / 2
    if wrist[1] > midpoint:
        state = "RESTING"
    elif state == "RESTING" and wrist[1] <= midpoint:
        state = "EXTENDED"
    elif state == "EXTENDED" and angle < 100:
        state = "FLEXED"
    elif state == "FLEXED" and angle > 160:
        state = "EXTENDED"
        rep_count += 1
    return state, rep_count


# EDA function
def perform_eda(csv_file):
    # Load the CSV into a DataFrame
    df = pd.read_csv(csv_file)

    # Display the first few rows of the dataframe
    st.subheader("Data Preview")
    st.write(df.head())

    # Show basic statistics
    st.subheader("Basic Statistics")
    st.write(df.describe())

    # Plot the angles for the left and right arms over time
    st.subheader("Arm Angles Over Time")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['Time'], df['Left_Angle'], label='Left Arm Angle')
    ax.plot(df['Time'], df['Right_Angle'], label='Right Arm Angle')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title('Arm Angles Over Time')
    ax.legend()

    # Render the plot in Streamlit
    st.pyplot(fig)

# Main function for the Streamlit app
def main():
    st.title("Shoulder Press Tracker and EDA")

    # File uploader for video
    uploaded_file = st.file_uploader("Upload an MP4 video", type=["mp4"])

    if uploaded_file is not None:
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        # Output paths for processed video and CSV
        output_video = "output_video.mp4"
        csv_output = "output_data.csv"

        # Run shoulder press tracking
        track_shoulder_presses_on_video(temp_file_path, output_video, csv_output)

        # Provide download buttons for the processed video and CSV file
        st.download_button("Download Processed Video", output_video)
        st.download_button("Download CSV Data", csv_output)

        # Automatically perform EDA on the generated CSV file
        perform_eda(csv_output)

if __name__ == "__main__":
    main()