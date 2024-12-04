from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

class ExerciseState:
    def __init__(self):
        self.counter = 0
        self.stage = None
        self.feedback = ""

def calculate_angle(a, b, c):
    """Calculate angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    return angle if angle <= 180 else 360 - angle

def process_frame(frame, exercise_type, state):
    """Process video frame to track movement and count reps."""
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    feedback = "Keep going!"

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        if exercise_type == "bicep_curl":
            # Extract joint coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)

            # Stage logic
            if angle > 160 and state.stage == "up":
                state.stage = "down"
            elif angle < 40 and state.stage == "down":
                state.stage = "up"
                state.counter += 1
                feedback = f"Good rep! Count: {state.counter}"

            # Draw feedback
            cv2.putText(frame, f"Angle: {int(angle)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        elif exercise_type == "shoulder_press":
            # Add shoulder press logic similar to above
            pass

    # Draw rep count and feedback
    cv2.putText(frame, f"Reps: {state.counter}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, feedback, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

# Initialize Exercise State
state = ExerciseState()

def generate_frames(exercise_type):
    """Generate video frames with movement detection."""
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Process the frame for the selected exercise
        processed_frame = process_frame(frame, exercise_type, state)

        # Encode the frame
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')

@app.route('/video_feed/<exercise_type>')
def video_feed(exercise_type):
    """Video streaming route."""
    return Response(generate_frames(exercise_type),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_count')
def get_count():
    """Get current rep count."""
    return jsonify({'count': state.counter, 'feedback': state.feedback})

@app.route('/reset_count')
def reset_count():
    """Reset rep counter."""
    state.counter = 0
    state.stage = None
    state.feedback = ""
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)
