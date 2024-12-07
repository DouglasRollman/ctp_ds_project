import cv2
import mediapipe as mp
import numpy as np
import time
from sklearn.ensemble import IsolationForest

# Mediapipe utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# Function to calculate angles between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    if angle > 180.0:
        angle = 360 - angle
    return angle


# Function to draw text with a background
def draw_text_with_background(image, text, position, font, font_scale, color, thickness, bg_color, padding=10):
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x, text_y = position
    box_coords = (
        (text_x - padding, text_y - padding),
        (text_x + text_size[0] + padding, text_y + text_size[1] + padding),
    )
    cv2.rectangle(image, box_coords[0], box_coords[1], bg_color, cv2.FILLED)
    cv2.putText(image, text, (text_x, text_y + text_size[1]), font, font_scale, color, thickness)


# Real-time feedback for single rep
def analyze_single_rep(rep, rep_data):
    """Provide actionable feedback for a single rep."""
    feedback = []
    avg_rom = np.mean([r["ROM"] for r in rep_data])
    avg_tempo = np.mean([r["Tempo"] for r in rep_data])
    avg_smoothness = np.mean([r["Smoothness"] for r in rep_data])

    if rep["ROM"] < avg_rom * 0.8:
        feedback.append("Extend arm more")
    if rep["Tempo"] < avg_tempo * 0.8:
        feedback.append("Slow down")
    if rep["Smoothness"] > avg_smoothness * 1.2:
        feedback.append("Move smoothly")

    return " | ".join(feedback) if feedback else "Good rep!"


# Post-workout feedback function with Isolation Forest
def analyze_workout_with_isolation_forest(rep_data):
    if not rep_data:
        print("No reps completed.")
        return

    print("\n--- Post-Workout Summary ---")

    # Convert rep_data to a feature matrix
    features = np.array([[rep["ROM"], rep["Tempo"], rep["Smoothness"]] for rep in rep_data])

    # Train Isolation Forest
    model = IsolationForest(contamination=0.2, random_state=42)
    predictions = model.fit_predict(features)

    # Analyze reps
    for i, (rep, prediction) in enumerate(zip(rep_data, predictions), 1):
        status = "Good" if prediction == 1 else "Anomalous"
        reason = []
        if prediction == -1:  # If anomalous
            if rep["ROM"] < np.mean(features[:, 0]) - np.std(features[:, 0]):
                reason.append("Low ROM")
            if rep["Tempo"] < np.mean(features[:, 1]) - np.std(features[:, 1]):
                reason.append("Too Fast")
            if rep["Smoothness"] > np.mean(features[:, 2]) + np.std(features[:, 2]):
                reason.append("Jerky Movement")
        reason_str = ", ".join(reason) if reason else "None"
        print(f"Rep {i}: {status} | ROM: {rep['ROM']:.2f}, Tempo: {rep['Tempo']:.2f}s, Smoothness: {rep['Smoothness']:.2f} | Reason: {reason_str}")


# Main workout tracking function
def main():
    cap = cv2.VideoCapture(0)
    counter = 0  # Rep counter
    stage = None  # Movement stage
    max_reps = 10
    rep_data = []  # Store metrics for each rep
    feedback = ""  # Real-time feedback for the video feed
    workout_start_time = None  # Timer start

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Initialize workout start time
            if workout_start_time is None:
                workout_start_time = time.time()

            # Timer
            elapsed_time = time.time() - workout_start_time
            timer_text = f"Timer: {int(elapsed_time)}s"

            # Convert frame to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            # Convert back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Check if pose landmarks are detected
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Extract key joints
                shoulder = [
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
                ]
                elbow = [
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y
                ]
                wrist = [
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
                ]

                # Check visibility of key joints
                visibility_threshold = 0.5
                if (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility < visibility_threshold or
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility < visibility_threshold or
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility < visibility_threshold):
                    draw_text_with_background(image, "Ensure all key joints are visible!", (50, 150),
                                              cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5, (0, 0, 255))
                    cv2.imshow('Workout Feedback', image)
                    continue  # Skip processing if joints are not visible

                # Calculate the angle
                angle = calculate_angle(shoulder, elbow, wrist)

                # Stage logic for counting reps
                if angle > 160 and stage != "down":
                    stage = "down"
                    start_time = time.time()  # Start timing for the rep
                    start_angle = angle  # Record the starting angle

                    # Stop the program if it's the 10th rep down stage
                    if counter == max_reps:
                        print("Workout complete at rep 10 (down stage)!")
                        break
                elif angle < 40 and stage == "down":
                    stage = "up"
                    counter += 1
                    end_time = time.time()  # End timing for the rep
                    end_angle = angle  # Record the ending angle

                    # Calculate rep metrics
                    rom = start_angle - end_angle  # Range of Motion
                    tempo = end_time - start_time  # Duration of the rep
                    smoothness = np.std([start_angle, end_angle])  # Dummy smoothness metric
                    rep_data.append({"ROM": rom, "Tempo": tempo, "Smoothness": smoothness})

                    # Analyze the rep using Isolation Forest
                    feedback = analyze_single_rep(rep_data[-1], rep_data)

                # Wireframe color based on form
                wireframe_color = (0, 255, 0) if stage == "up" or stage == "down" else (0, 0, 255)

                # Draw wireframe
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=wireframe_color, thickness=5, circle_radius=4),
                    mp_drawing.DrawingSpec(color=wireframe_color, thickness=5, circle_radius=4)
                )

                # Display reps, stage, timer, and feedback
                draw_text_with_background(image, f"Reps: {counter}", (50, 150),
                                          cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5, (0, 0, 0))
                draw_text_with_background(image, f"Stage: {stage if stage else 'N/A'}", (50, 300),
                                          cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5, (0, 0, 0))
                draw_text_with_background(image, timer_text, (1000, 50),  # Timer in the top-right corner
                                          cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, (0, 0, 0))
                draw_text_with_background(image, feedback, (50, 450),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, (0, 0, 0))

            # Show video feed
            cv2.imshow('Workout Feedback', image)

            # Break if 'q' is pressed
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    # Post-workout analysis
    analyze_workout_with_isolation_forest(rep_data)


if __name__ == "__main__":
    main()
