import cv2
import json
import numpy as np

# Load keypoints from JSON
with open('video_skeleton_data.json', 'r') as f:
    loaded_keypoints_list = json.load(f)

# Load your avatar's images (e.g., head, body, limbs)
head_img = cv2.imread('head.png', cv2.IMREAD_UNCHANGED)
left_arm_img = cv2.imread('left_arm.png', cv2.IMREAD_UNCHANGED)
right_arm_img = cv2.imread('right_arm.png', cv2.IMREAD_UNCHANGED)
body_img = cv2.imread('body.png', cv2.IMREAD_UNCHANGED)
left_leg_img = cv2.imread('left_leg.png', cv2.IMREAD_UNCHANGED)
right_leg_img = cv2.imread('right_leg.png', cv2.IMREAD_UNCHANGED)

# Define keypoint indices for different body parts
keypoint_indices = {
    'head': 0,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Create a window to replay the movements
window_name = 'Avatar Replay'
cv2.namedWindow(window_name)

# Function to overlay transparent image (with alpha channel) on background
def overlay_image(bg_img, overlay_img, pos):
    x, y = pos
    h, w, _ = overlay_img.shape
    # Get the region of interest in the background
    roi = bg_img[y:y+h, x:x+w]
    
    # Masking the overlay image using alpha channel
    overlay_gray = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2BGRA)
    _, mask = cv2.threshold(overlay_gray[:, :, 3], 1, 255, cv2.THRESH_BINARY)
    inv_mask = cv2.bitwise_not(mask)

    # Make area where we will place the sprite transparent in background
    bg_img[y:y+h, x:x+w] = cv2.bitwise_and(roi, roi, mask=inv_mask)

    # Add the sprite to the frame
    bg_img[y:y+h, x:x+w] = cv2.add(bg_img[y:y+h, x:x+w], overlay_img)

# Replay the keypoints data and animate the character
for keypoints in loaded_keypoints_list:
    # Create a blank frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Scale the keypoints to fit the window size
    height, width = frame.shape[:2]
    points = []
    for i, (x, y, z, visibility) in enumerate(keypoints):
        if visibility > 0.5:  # Only use keypoints with good visibility
            cx, cy = int(x * width), int(y * height)
            points.append((cx, cy))
        else:
            points.append(None)  # Use None for invisible keypoints

    # Draw avatar parts (e.g., head, arms, body)
    if points[keypoint_indices['head']]:
        overlay_image(frame, head_img, points[keypoint_indices['head']])

    if points[keypoint_indices['left_shoulder']] and points[keypoint_indices['left_wrist']]:
        overlay_image(frame, left_arm_img, points[keypoint_indices['left_shoulder']])
    
    if points[keypoint_indices['right_shoulder']] and points[keypoint_indices['right_wrist']]:
        overlay_image(frame, right_arm_img, points[keypoint_indices['right_shoulder']])

    if points[keypoint_indices['left_hip']] and points[keypoint_indices['left_ankle']]:
        overlay_image(frame, left_leg_img, points[keypoint_indices['left_hip']])

    if points[keypoint_indices['right_hip']] and points[keypoint_indices['right_ankle']]:
        overlay_image(frame, right_leg_img, points[keypoint_indices['right_hip']])
    
    # Show the frame
    cv2.imshow(window_name, frame)

    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()