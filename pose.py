import cv2
import mediapipe as mp
import json
import csv
import numpy as np
import pandas as pd


def cam2csv(label, save_location):
    """
    Captures skeleton keypoints from a webcam video using MediaPipe's BlazePose and saves it to a CSV file.

    Parameters:
    - label (str): Label for the video capture.
    - save_location (str): Path to save the CSV file (e.g., 'file.csv').

    Return:
    - None: The function captures video from the webcam and saves the keypoints to the CSV file.
    """
    
    # Initialize MediaPipe BlazePose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Capture video from webcam
    cap = cv2.VideoCapture(0)

    # Store keypoints over time
    keypoints_list = []

    # Define CSV file headers (x, y, z, visibility for each of the 33 keypoints)
    headers = []
    for i in range(33):  # Assuming 33 keypoints detected by BlazePose
        headers.extend([f"x_{i}", f"y_{i}", f"z_{i}", f"visibility_{i}"])

    # Open CSV file for writing
    with open(save_location, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(headers)  # Write the header row

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Convert frame to RGB for MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False

            # Run BlazePose
            results = pose.process(image_rgb)

            # Convert back to BGR for OpenCV
            image_rgb.flags.writeable = True
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # If landmarks are detected
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Extract and store the keypoints (x, y, z) and visibility for each joint
                keypoints = []
                for landmark in landmarks:
                    keypoints.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])

                # Append the flattened keypoints to the CSV file
                csv_writer.writerow(keypoints)

                # Draw the pose landmarks on the frame (for visualization)
                mp.solutions.drawing_utils.draw_landmarks(
                    image_bgr, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS
                )

            # Show the frame with pose landmarks
            cv2.imshow(label, image_bgr)

            # Press q to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Example usage:
# capture_skeleton_data('Skeleton Capture', 'skeleton_data.csv')

def cam2json(label, save_location):
    """
    Captures skeleton keypoints from a webcam video using MediaPipe's BlazePose and saves it to a JSON file.

    Parameters:
    - label (str): Label for the video capture.
    - save_location (str): Path to save the JSON file (e.g., 'file.json').

    Return:
    - None: The function captures video from the webcam and saves the keypoints to the JSON file.
    """

    # Initialize MediaPipe BlazePose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Capture video from webcam
    cap = cv2.VideoCapture(0)

    # Store keypoints over time
    keypoints_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert frame to RGB for MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        # Run BlazePose
        results = pose.process(image_rgb)

        # Convert back to BGR for OpenCV
        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # If landmarks are detected
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Store keypoints for the current frame
            keypoints = {
                "frame": len(keypoints_data),  # Frame number
                "keypoints": []
            }
            for landmark in landmarks:
                keypoints["keypoints"].append({
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                    "visibility": landmark.visibility
                })

            # Add the keypoints data for this frame to the main list
            keypoints_data.append(keypoints)

            # Draw the pose landmarks on the frame (for visualization)
            mp.solutions.drawing_utils.draw_landmarks(
                image_bgr, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS
            )

        # Show the frame with pose landmarks
        cv2.imshow(label, image_bgr)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Save the keypoints data to a JSON file
    with open(save_location, 'w') as jsonfile:
        json.dump(keypoints_data, jsonfile, indent=4)

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Example usage:
# cam2json('Skeleton Capture', 'skeleton_data.json')

def video_keypoints_json(video_paths, exercise_label, json_file):
    """
    Processes a list of videos using MediaPipe's BlazePose, extracts keypoints, 
    and saves them into a JSON file with an exercise label and timestamps.

    Parameters:
    - video_paths (list of str): List of file paths to the videos you want to process.
    - exercise_label (str): Label to identify the type of exercise (e.g., 'pushup', 'squat').
                            This label will be attached to each entry in the JSON file.
    - json_file (str): Name (and path) of the JSON file where the keypoints will be saved.

    The JSON file will contain a list of dictionaries, each with the following keys:
    - label: The exercise type
    - timestamp: The time of the frame in seconds
    - keypoints: A list of x, y, z coordinates and visibility scores for each of the 33 detected keypoints.

    Return:
    - None: The function writes the extracted keypoints to a JSON file but doesn't return anything.
    """
    
    # Initialize MediaPipe BlazePose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.0, min_tracking_confidence=0.0)

    # Initialize a list to hold the results
    results_list = []

    # Process each video in the video_paths list
    for video_path in video_paths:
        # Load the video from the specified path
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Unable to open video {video_path}")
            continue  # Skip to the next video if the current one fails to open

        # Process the video frame by frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Exit the loop if no frames are left

            # Get the current timestamp (in seconds)
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Convert milliseconds to seconds

            # Convert the image to RGB for MediaPipe processing
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False  # Improve performance by disabling image writeability

            # Run BlazePose on the image to detect pose landmarks
            results = pose.process(image_rgb)

            # If landmarks are detected in the image
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark  # Get the landmark data

                # Extract and flatten the keypoints (x, y, z, visibility) for each joint (keypoint)
                keypoints = []
                for landmark in landmarks:
                    keypoints.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])

                # Create a dictionary for the current frame's data
                frame_data = {
                    "label": exercise_label,
                    "timestamp": timestamp,
                    "keypoints": keypoints
                }

                # Add the frame data to the results list
                results_list.append(frame_data)
            else:
                # If no keypoints were detected, print a message
                print(f"No keypoints detected at timestamp {timestamp} for video {video_path}.")

        cap.release()  # Release the video capture object after processing

    # Clean up the pose instance after processing is done
    pose.close()

    # Write the results list to a JSON file
    with open(json_file, 'w') as jsonfile:
        json.dump(results_list, jsonfile, indent=4)

    # Print a message indicating that the keypoints have been successfully saved
    print(f"Keypoints from {len(video_paths)} videos labeled with '{exercise_label}' saved to {json_file}.")

# Example usage
# video_paths = ['data/pushups/video1.mp4', 'data/pushups/video2.mp4']
# exercise_label = 'pushup'
# json_file = 'video_keypoints.json'
# video_keypoints_json(video_paths, exercise_label, json_file)

def video_keypoints_csv(video_paths, exercise_label, csv_file):
    """
    Processes a list of videos using MediaPipe's BlazePose, extracts keypoints, 
    and saves them into a CSV file with an exercise label and timestamps.

    Parameters:
    - video_paths (list of str): List of file paths to the videos you want to process.
    - exercise_label (str): Label to identify the type of exercise (e.g., 'pushup', 'squat').
                            This label will be attached to each row of keypoints in the CSV file.
    - csv_file (str): Name (and path) of the CSV file where the keypoints will be saved.

    The CSV file will contain the following columns:
    - label: The exercise type
    - timestamp: The time of the frame in seconds
    - x_n, y_n, z_n, visibility_n: The x, y, z coordinates and visibility score for each of the 33 detected keypoints,
                                   where `n` represents the index of the keypoint.

    Return:
    - None: The function writes the extracted keypoints to a CSV file but doesn't return anything.
    """

    # Initialize MediaPipe BlazePose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)  # Adjust confidence thresholds if needed

    # Define headers for CSV: x, y, z, visibility for each of the 33 keypoints, and an exercise label
    headers = ['label', 'timestamp']  # Start with the label and timestamp header
    for i in range(33):  # BlazePose detects 33 keypoints
        headers.extend([f"x_{i}", f"y_{i}", f"z_{i}", f"visibility_{i}"])

    # Open CSV file for writing the results
    with open(csv_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(headers)  # Write the header row with label and keypoints

        # Process each video in the video_paths list
        for video_path in video_paths:
            # Load the video from the specified path
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Unable to open video {video_path}")
                continue  # Skip to the next video if the current one fails to open

            print(f"Processing video: {video_path}")

            # Process the video frame by frame
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break  # Exit the loop if no frames are left

                # Get the current timestamp (in seconds)
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Convert milliseconds to seconds

                # Convert the image to RGB for MediaPipe processing
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False  # Improve performance by disabling image writeability

                # Run BlazePose on the image to detect pose landmarks
                results = pose.process(image_rgb)

                # If landmarks are detected in the image
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark  # Get the landmark data

                    # Extract and flatten the keypoints (x, y, z, visibility) for each joint (keypoint)
                    keypoints = []
                    for landmark in landmarks:
                        keypoints.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])

                    # Prepend the exercise label and timestamp to the keypoints data and write the row to the CSV file
                    csv_writer.writerow([exercise_label, timestamp] + keypoints)
                else:
                    # If no keypoints were detected, optionally log or handle this case
                    print(f"No keypoints detected at timestamp {timestamp} for video {video_path}.")

            cap.release()  # Release the video capture object after processing

    # Clean up the pose instance after processing is done
    pose.close()

    # Print a message indicating that the keypoints have been successfully saved
    print(f"Keypoints from {len(video_paths)} videos labeled with '{exercise_label}' saved to {csv_file}.")

# Example usage:
# video_keypoints_csv(['video1.mp4', 'video2.mp4'], 'pushup', 'output_keypoints.csv')



def img_keypoints_csv(image_paths, exercise_label, csv_file):
    """
    Processes a list of images using MediaPipe's BlazePose, extracts keypoints, 
    and saves them into a CSV file with an exercise label.

    Parameters:
    - image_paths (list of str): List of file paths to the images you want to process.
    - exercise_label (str): Label to identify the type of exercise (e.g., 'pushup', 'squat').
                            This label will be attached to each row of keypoints in the CSV file.
    - csv_file (str): Name (and path) of the CSV file where the keypoints will be saved.

    The CSV file will contain the following columns:
    - label: The exercise type
    - x_n, y_n, z_n, visibility_n: The x, y, z coordinates and visibility score for each of the 33 detected keypoints,
                                   where `n` represents the index of the keypoint.

    Return:
    - None: The function writes the extracted keypoints to a CSV file but doesn't return anything.
    """
    
    # Initialize MediaPipe BlazePose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.1, min_tracking_confidence=0.1)

    # Define headers for CSV: x, y, z, visibility for each of the 33 keypoints, and an exercise label
    headers = ['label']  # Start with the label header for exercise type
    for i in range(33):  # BlazePose detects 33 keypoints
        headers.extend([f"x_{i}", f"y_{i}", f"z_{i}", f"visibility_{i}"])

    # Open CSV file for writing the results
    with open(csv_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(headers)  # Write the header row with label and keypoints

        # Process each image in the image_paths list
        for image_path in image_paths:
            # Load the image from the specified path
            image = cv2.imread(image_path)
            
            # Check if the image was successfully loaded
            if image is None:
                print(f"Error: Unable to load image {image_path}")
                continue  # Skip to the next image if the current one fails to load

            # Convert the image to RGB for MediaPipe processing
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False  # Improve performance by disabling image writeability

            # Run BlazePose on the image to detect pose landmarks
            results = pose.process(image_rgb)

            # If landmarks are detected in the image
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark  # Get the landmark data

                # Extract and flatten the keypoints (x, y, z, visibility) for each joint (keypoint)
                keypoints = []
                for landmark in landmarks:
                    keypoints.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])

                # Prepend the exercise label to the keypoints data and write the row to the CSV file
                csv_writer.writerow([exercise_label] + keypoints)
            else:
                # If no keypoints were detected, print a message
                print(f"No keypoints detected for image {image_path}.")

    # Clean up the pose instance after processing is done
    pose.close()

    # Print a message indicating that the keypoints have been successfully saved
    print(f"Keypoints from {len(image_paths)} images labeled with '{exercise_label}' saved to {csv_file}.")

def img_keypoints_json(image_paths, exercise_label, json_file):
    """
    Processes a list of images using MediaPipe's BlazePose, extracts keypoints, 
    and saves them into a JSON file with an exercise label.

    Parameters:
    - image_paths (list of str): List of file paths to the images you want to process.
    - exercise_label (str): Label to identify the type of exercise (e.g., 'pushup', 'squat').
                            This label will be attached to each entry of keypoints in the JSON file.
    - json_file (str): Name (and path) of the JSON file where the keypoints will be saved.

    The JSON file will contain an array of objects, each having the following structure:
    - label: The exercise type
    - keypoints: Array of keypoint objects, each containing:
      - x: x coordinate
      - y: y coordinate
      - z: z coordinate
      - visibility: visibility score

    Return:
    - None: The function writes the extracted keypoints to a JSON file but doesn't return anything.
    """
    
    # Initialize MediaPipe BlazePose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.0, min_tracking_confidence=0.0)

    # Prepare a list to hold the keypoints data
    keypoints_data = []

    # Process each image in the image_paths list
    for image_path in image_paths:
        # Load the image from the specified path
        image = cv2.imread(image_path)
        
        # Check if the image was successfully loaded
        if image is None:
            print(f"Error: Unable to load image {image_path}")
            continue  # Skip to the next image if the current one fails to load

        # Convert the image to RGB for MediaPipe processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False  # Improve performance by disabling image writeability

        # Run BlazePose on the image to detect pose landmarks
        results = pose.process(image_rgb)

        # If landmarks are detected in the image
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark  # Get the landmark data

            # Extract keypoints (x, y, z, visibility) for each joint (keypoint)
            keypoints = []
            for landmark in landmarks:
                keypoints.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })

            # Append the exercise label and keypoints data to the list
            keypoints_data.append({
                'label': exercise_label,
                'keypoints': keypoints
            })
        else:
            # If no keypoints were detected, print a message
            print(f"No keypoints detected for image {image_path}.")

    # Write the collected keypoints data to a JSON file
    with open(json_file, 'w') as jsonfile:
        json.dump(keypoints_data, jsonfile, indent=4)

    # Clean up the pose instance after processing is done
    pose.close()

    # Print a message indicating that the keypoints have been successfully saved
    print(f"Keypoints from {len(image_paths)} images labeled with '{exercise_label}' saved to {json_file}.")

#########################################################################################################################
"""
Visualizing Pose Data via OpenCV and MediaPipe
"""
#########################################################################################################################
# Define keypoint connections (from BlazePose)
# Global variable for keypoint connections do not delete or change
keypoint_connections = [
    (0,4), (0,1),
    (6, 5), (4, 5), (6, 8),
    (1, 2), (2, 3), (3, 7),
    (10, 9),
    (20, 18), (18, 16), (16, 22), (16, 20),
    (16, 14), (14, 12), (12, 11), (11, 13),
    (13, 15), (15, 21), (15, 19), (15, 17), (19, 17),
    (12, 24), (11, 23), (24, 23),
    (24, 26), (26, 28), (32, 28), (30, 28), (32, 30),
    (23, 25), (27, 25), (27, 29), (29, 31), (27, 31)
]
#########################################################################################################################

def play_json(json_file):
    """
    Replays skeleton movements from a JSON file containing keypoints data.

    Parameters:
    - json_file (str): Path to the JSON file containing keypoints data.
    
    Return:
    - None: The function displays the skeleton movements in a window.
    """
    
    # Load keypoints from JSON
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Create a window to replay the movements
    window_name = 'Skeleton Replay'
    cv2.namedWindow(window_name)

    # Iterate through each frame's keypoints in the JSON data
    for frame_data in data:
        keypoints_list = frame_data['keypoints']  # Extract the keypoints

        # Create an empty frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Scale the keypoints to fit the window size
        height, width = frame.shape[:2]
        scaled_keypoints = []

        # Process the flattened keypoints list into groups of 4 (x, y, z, visibility)
        for i in range(0, len(keypoints_list), 4):
            x, y, z, visibility = keypoints_list[i:i+4]  # Get 4 values for each keypoint
            if visibility > 0.5:  # Only show keypoints with good visibility
                cx, cy = int(x * width), int(y * height)
                scaled_keypoints.append((cx, cy))  # Store for drawing lines
                # Draw keypoint as a small circle
                cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
            else:
                scaled_keypoints.append(None)  # If not visible, mark as None

        # Draw the skeleton by connecting keypoints
        for start_idx, end_idx in keypoint_connections:
            if scaled_keypoints[start_idx] is not None and scaled_keypoints[end_idx] is not None:
                cv2.line(frame, scaled_keypoints[start_idx], scaled_keypoints[end_idx], (0, 255, 0), 2)

        # Show the frame
        cv2.imshow(window_name, frame)

        # Wait for a short period to create a video effect
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# Example usage:
# play_json('dance.json')


def play_csv(csv_file):
    """
    Replays skeleton movements from a CSV file containing keypoints data.

    Parameters:
    - csv_file (str): Path to the CSV file containing keypoints data.

    Return:
    - None: The function displays the skeleton movements in a window.
    """
    # Load keypoints from CSV
    keypoints_data = pd.read_csv(csv_file)

    # Create a window to replay the movements
    window_name = 'Skeleton Replay'
    cv2.namedWindow(window_name)

    # Replay the keypoints data
    for index, row in keypoints_data.iterrows():
        keypoints = []

        # Extract keypoints from the row
        for i in range(33):  # Assuming 33 keypoints
            keypoints.append({
                'x': row[f'x_{i}'],
                'y': row[f'y_{i}'],
                'z': row[f'z_{i}'],
                'visibility': row[f'visibility_{i}']
            })

        frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Create a blank frame

        # Scale the keypoints to fit the window size
        height, width = frame.shape[:2]
        scaled_keypoints = []
        for landmark in keypoints:
            if landmark['visibility'] > 0.4:  # Only show keypoints with good visibility
                cx, cy = int(landmark['x'] * width), int(landmark['y'] * height)
                scaled_keypoints.append((cx, cy))  # Store for drawing lines
                # Draw keypoint as a small circle
                cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
            else:
                scaled_keypoints.append(None)  # If not visible, mark as None

        # Draw the skeleton by connecting keypoints
        for start_idx, end_idx in keypoint_connections:
            if scaled_keypoints[start_idx] is not None and scaled_keypoints[end_idx] is not None:
                cv2.line(frame, scaled_keypoints[start_idx], scaled_keypoints[end_idx], (0, 255, 0), 2)

        # Show the frame
        cv2.imshow(window_name, frame)

        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# Example usage
# play_csv('skeleton_data.csv')


def display_images(image_paths):
    """
    Displays a list of images in a window one by one.

    Parameters:
    - image_paths (list of str): List of file paths to the images you want to display.

    Return:
    - None: The function displays each image in a window and waits for a key press to move to the next image.
    """
    
    for image_path in image_paths:
        # Load the image from the specified path
        image = cv2.imread(image_path)
        
        # Check if the image was successfully loaded
        if image is None:
            print(f"Error: Unable to load image {image_path}")
            continue  # Skip to the next image if the current one fails to load
        
        # Display the image in a window
        cv2.imshow('Image Display', image)

        # Wait for a key press to move to the next image
        key = cv2.waitKey(0)  # Wait indefinitely until a key is pressed
        if key == ord('q'):  # Quit if 'q' is pressed
            break

    cv2.destroyAllWindows()

# Example usage
# image_list = ['image1.jpg', 'image2.jpg', 'image3.jpg']
# display_images(image_list)

def display_keypoints_imgs(df):
    """
    Displays images with keypoints from a DataFrame row by row.

    Parameters:
    - df (pd.DataFrame): DataFrame where each row represents an image and contains
                         the x, y, z coordinates and visibility scores for keypoints.

    Return:
    - None: The function displays each set of keypoints on a blank canvas.
    """

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        # Create a blank frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Extract keypoints from the row
        keypoints = []
        for i in range(33):  # Assuming there are 33 keypoints
            x = row[f'x_{i}']
            y = row[f'y_{i}']
            z = row[f'z_{i}']
            visibility = row[f'visibility_{i}']
            keypoints.append((x, y, z, visibility))

        # Scale the keypoints to fit the window size (assuming normalized coordinates)
        height, width = frame.shape[:2]
        scaled_keypoints = []
        for x, y, z, visibility in keypoints:
            if visibility > 0.4:  # Only show keypoints with good visibility
                cx, cy = int(x * width), int(y * height)
                scaled_keypoints.append((cx, cy))
                # Draw keypoint as a small circle
                cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1)
            else:
                scaled_keypoints.append(None)  # If not visible, mark as None

        # Draw the skeleton by connecting keypoints
        for start_idx, end_idx in keypoint_connections:
            if scaled_keypoints[start_idx] is not None and scaled_keypoints[end_idx] is not None:
                cv2.line(frame, scaled_keypoints[start_idx], scaled_keypoints[end_idx], (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Keypoint Display', frame)

        # Wait for a key press to move to the next image
        key = cv2.waitKey(0)  # Wait indefinitely until a key is pressed
        if key == ord('q'):  # Quit if 'q' is pressed
            break

    cv2.destroyAllWindows()

# Example usage
# Assuming your DataFrame is structured as follows
# df = pd.DataFrame({
#     'x_0': [...], 'y_0': [...], 'z_0': [...], 'visibility_0': [...],
#     'x_1': [...], 'y_1': [...], 'z_1': [...], 'visibility_1': [...],
#     ...
# })
# display_keypoints(df)