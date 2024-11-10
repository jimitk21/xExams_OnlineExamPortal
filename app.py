from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import mediapipe as mp
import base64
import numpy as np
from PIL import Image
import io
import logging
import json

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize MediaPipe Face Detection and Face Mesh
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Initialize the face detection and face mesh models
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.2, min_tracking_confidence=0.2)

# Variables to monitor head movement
previous_face_location = None

# Face Mesh landmark indices for eyes
LEFT_EYE_LANDMARKS = list(range(33, 133))
RIGHT_EYE_LANDMARKS = list(range(362, 463))

def process_frame(base64_frame):
    try:
        # Decode base64 image
        encoded_data = base64_frame.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with face detection
        detection_results = face_detection.process(rgb_frame)
        mesh_results = face_mesh.process(rgb_frame)
        
        warnings = []
        global previous_face_location
        
        # Process face detection results
        if detection_results.detections:
            # Check for multiple faces
            if len(detection_results.detections) > 1:
                warnings.append("Multiple faces detected")
            
            # Process first detected face
            detection = detection_results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            current_location = (bbox.xmin, bbox.ymin, bbox.width, bbox.height)
            
            # Check head movement
            if previous_face_location:
                movement = sum(abs(c - p) for c, p in zip(current_location, previous_face_location))
                if movement > 0.1:
                    warnings.append("Excessive head movement detected")
            
            previous_face_location = current_location
        else:
            warnings.append("No face detected")
        
        # Process face mesh for eye tracking
        if mesh_results.multi_face_landmarks:
            face_landmarks = mesh_results.multi_face_landmarks[0]
            
            # Calculate eye positions
            left_eye = [face_landmarks.landmark[i] for i in LEFT_EYE_LANDMARKS]
            right_eye = [face_landmarks.landmark[i] for i in RIGHT_EYE_LANDMARKS]
            
            left_eye_x = sum(landmark.x for landmark in left_eye) / len(left_eye)
            right_eye_x = sum(landmark.x for landmark in right_eye) / len(right_eye)
            
            # Check if looking away
            if left_eye_x < 0.35 or right_eye_x > 0.65:
                warnings.append("Looking away detected")
        
        return warnings
        
    except Exception as e:
        logging.error(f"Error processing frame: {str(e)}")
        return ["Error processing video frame"]

@app.route('/')
def index():
    return render_template('exam.html')

@socketio.on('process_frame')
def handle_frame(data):
    warnings = process_frame(data['frame'])
    emit('processing_result', {'warnings': warnings})

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000)