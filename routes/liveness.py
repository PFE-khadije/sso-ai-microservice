import cv2
import numpy as np
import mediapipe as mp
from flask import Blueprint, request, jsonify
from services.image_utils import decode_image_bytes

liveness_bp = Blueprint('liveness', __name__)

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

def eye_aspect_ratio(landmarks, eye_indices):
    """
    Calculate Eye Aspect Ratio (EAR) for a given eye.
    Indices are from MediaPipe's 468-landmark model.
    """
    points = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_indices])
    # Vertical distances
    A = np.linalg.norm(points[1] - points[5])
    B = np.linalg.norm(points[2] - points[4])
    # Horizontal distance
    C = np.linalg.norm(points[0] - points[3])
    ear = (A + B) / (2.0 * C)
    return ear

@liveness_bp.route('/liveness', methods=['POST'])
def liveness_check():
    """
    Expects a sequence of images (frames) as multipart files: frame0, frame1, frame2, ...
    Returns: { "liveness": bool, "confidence": float }
    """
    # Collect frames
    frames = []
    i = 0
    while f"frame{i}" in request.files:
        frame_bytes = request.files[f"frame{i}"].read()
        img = decode_image_bytes(frame_bytes)
        if img is not None:
            frames.append(img)
        i += 1

    if len(frames) < 3:
        return jsonify({"error": "At least 3 frames are required"}), 400

    ear_values = []
    for frame in frames:
        # Convert BGR to RGB (MediaPipe expects RGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            # Indices for left and right eyes (from MediaPipe)
            LEFT_EYE = [33, 160, 158, 133, 153, 144]
            RIGHT_EYE = [362, 385, 387, 263, 373, 380]
            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
            ear = (left_ear + right_ear) / 2.0
            ear_values.append(ear)
        else:
            # If face is lost in any frame, liveness fails
            return jsonify({"liveness": False, "confidence": 0.0, "error": "Face not detected in all frames"}), 400

    if len(ear_values) < 3:
        return jsonify({"liveness": False, "confidence": 0.0, "error": "Not enough valid frames"}), 400

    # Detect a blink: a sudden drop in EAR followed by recovery
    # Calculate baseline (average of first and last frame)
    baseline = (ear_values[0] + ear_values[-1]) / 2
    threshold = baseline * 0.7  # blink threshold: 30% drop

    blink_detected = False
    for i in range(1, len(ear_values)-1):
        if ear_values[i] < threshold and ear_values[i] < ear_values[i-1] and ear_values[i] < ear_values[i+1]:
            blink_detected = True
            break

    confidence = 0.8 if blink_detected else 0.2
    return jsonify({"liveness": blink_detected, "confidence": confidence})
