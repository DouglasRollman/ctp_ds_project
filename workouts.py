import cv2
import mediapipe as mp
import math

# Initialize mediapipe Pose module
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    # a, b, c are points represented as (x, y)
    ab = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]
    # Calculate the cosine of the angle
    dot_product = ab[0] * bc[0] + ab[1] * bc[1]
    mag_ab = math.sqrt(ab[0]**2 + ab[1]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)
    cos_angle = dot_product / (mag_ab * mag_bc)
    angle = math.acos(cos_angle) * 180.0 / math.pi
    return angle

# Function to track shoulder presses for both arms
def track_shoulder_presses(video_source=0):
    # Video feed
    cap = cv2.VideoCapture(video_source)  # Change to 0, 1, or -1 based on your webcam

    # Mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        left_rep_count = 0
        right_rep_count = 0
        left_is_pressing = False
        right_is_pressing = False
        left_prev_angle = None
        right_prev_angle = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

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
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * frame.shape[1],
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * frame.shape[0]]

                # Get coordinates for right arm
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1],
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * frame.shape[1],
                               landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * frame.shape[0]]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * frame.shape[1],
                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * frame.shape[0]]

                # Calculate angles for both elbows
                left_angle = calculate_angle(left_shoulder, left_elbow, left_hip)
                right_angle = calculate_angle(right_shoulder, right_elbow, right_hip)

                # Detect shoulder press motion for left arm
                if left_prev_angle and left_angle < 100 and left_prev_angle > 160:  # Transition from full extension to flexion
                    if not left_is_pressing:
                        left_rep_count += 1
                        left_is_pressing = True
                elif left_angle > 160:  # Fully extended position
                    left_is_pressing = False
                left_prev_angle = left_angle

                # Detect shoulder press motion for right arm
                if right_prev_angle and right_angle < 100 and right_prev_angle > 160:  # Transition from full extension to flexion
                    if not right_is_pressing:
                        right_rep_count += 1
                        right_is_pressing = True
                elif right_angle > 160:  # Fully extended position
                    right_is_pressing = False
                right_prev_angle = right_angle

                # Display angles near the shoulders
                cv2.putText(image, f'{int(left_angle)}°',
                            (int(left_shoulder[0]), int(left_shoulder[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(image, f'{int(right_angle)}°',
                            (int(right_shoulder[0]), int(right_shoulder[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Display rep counts
                cv2.putText(image, f'Left Reps: {left_rep_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(image, f'Right Reps: {right_rep_count}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Render detections
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            # Display frame
            cv2.imshow('Shoulder Press Tracker', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'Q' to exit
                break

        cap.release()
        cv2.destroyAllWindows()

# Run the tracker
track_shoulder_presses()