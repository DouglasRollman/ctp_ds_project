import cv2
import mediapipe as mp
import json
import time

# Initialize MediaPipe BlazePose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load video file
video_path = 'breakdance.mp4'
cap = cv2.VideoCapture(video_path)

# Get total number of frames and the frame rate of the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Estimate the total video duration in seconds
total_duration = total_frames / fps
print(f"Total video duration: {total_duration / 60:.2f} minutes")

# Store keypoints over time
keypoints_list = []

# Track time
start_time = time.time()

frame_counter = 0  # Initialize frame counter

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video has ended.")
        break

    # Increment frame counter
    frame_counter += 1

    # Convert frame to RGB for MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False

    # Run BlazePose
    results = pose.process(image_rgb)

    # If landmarks are detected
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Extract and store the keypoints (x, y, z) and visibility for each joint
        keypoints = []
        for landmark in landmarks:
            keypoints.append((landmark.x, landmark.y, landmark.z, landmark.visibility))

        keypoints_list.append(keypoints)

    # Estimate elapsed time and remaining time
    elapsed_time = time.time() - start_time
    progress = frame_counter / total_frames
    estimated_total_time = elapsed_time / progress if progress > 0 else 0
    remaining_time = estimated_total_time - elapsed_time

    # Print progress and estimated time left
    print(f"Processed {frame_counter}/{total_frames} frames "
          f"({progress * 100:.2f}% complete), "
          f"Elapsed time: {elapsed_time:.2f}s, "
          f"Estimated time left: {remaining_time:.2f}s")

    # Press 'q' to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save keypoints_list to a JSON file
with open('breakdance_data.json', 'w') as f:
    json.dump(keypoints_list, f)

cap.release()
cv2.destroyAllWindows()