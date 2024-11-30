import cv2
import mediapipe as mp
import numpy as np
import pickle
import base64
import sqlite3
from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import os
import time

app = Flask(__name__)
CORS(app)

# Initialize database connection
conn = sqlite3.connect("facial_data.db", check_same_thread=False)
cursor = conn.cursor()

def save_image_with_landmarks(image, landmarks, output_path):
    """
    Draw landmarks on the image and save it to the specified path.
    :param image: A NumPy array representing the image.
    :param landmarks: A NumPy array of facial landmarks.
    :param output_path: Path to save the annotated image.
    """
    for (x, y, _) in landmarks:
        # Convert normalized coordinates to pixel coordinates
        h, w, _ = image.shape
        px, py = int(x * w), int(y * h)
        cv2.circle(image, (px, py), 2, (0, 255, 0), -1)  # Draw a green dot for each landmark
    
    # Save the image
    cv2.imwrite(output_path, image)

def validate_base64(data):
    base64_pattern = r'^[A-Za-z0-9+/]+={0,2}$'
    # Remove any newlines or whitespace
    data = data.strip()
    # Check if the data length is correct
    if len(data) % 4 != 0:
        return False
    # Check if the data matches the Base64 pattern
    return re.match(base64_pattern, data) is not None

def extract_facial_landmarks(image):
    """
    Extract facial landmarks from an image using MediaPipe.
    :param image: A NumPy array representing the image.
    :return: A NumPy array of facial landmarks.
    """
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

    # Convert the image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image to detect facial landmarks
    results = face_mesh.process(rgb_image)

    landmarks = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                landmarks.append((landmark.x, landmark.y, landmark.z))

    face_mesh.close()
    return np.array(landmarks)

def save_landmarks(patient_id, landmarks):
    serialized_landmarks = pickle.dumps(landmarks)
    cursor.execute('''
    INSERT OR REPLACE INTO patient_facial_data (patient_id, landmarks)
    VALUES (?, ?)
    ''', (patient_id, serialized_landmarks))
    conn.commit()

def fetch_landmarks(patient_id):
    cursor.execute('''
    SELECT landmarks FROM patient_facial_data WHERE patient_id = ?
    ''', (patient_id,))
    row = cursor.fetchone()
    if row:
        return pickle.loads(row[0])
    return None

def check_significant_changes(current_landmarks, previous_landmarks, threshold=0.1):
    # Calculate the differences in landmarks
    changes = np.linalg.norm(current_landmarks - previous_landmarks, axis=1)
    average_change = np.mean(changes)

    # Check if the average change exceeds the threshold
    if average_change > threshold:
        return True, average_change
    return False, average_change

def run_comparison(image_data, patient_id):
    """
    Compare the facial landmarks of the given image with the previously saved landmarks.
    Save the image with landmarks annotated and return comparison results.
    :param image_data: A NumPy array representing the image.
    :param patient_id: The patient's unique identifier.
    :return: A dictionary with comparison results and the Base64-encoded image.
    """
    # Extract landmarks from the current image
    current_landmarks = extract_facial_landmarks(image_data)

    if len(current_landmarks) == 0:
        return {"error": "No face detected in the image."}

    # Fetch previous landmarks from the database
    significant_change = False
    change_value = 0.0
    previous_landmarks = fetch_landmarks(patient_id)

    if previous_landmarks is not None:
        # Compare current and previous landmarks
        significant_change, change_value = check_significant_changes(
            current_landmarks, previous_landmarks
        )

    # Save the current landmarks for future comparisons
    save_landmarks(patient_id, current_landmarks)

    # Save the annotated image
    images_dir = "./images"
    os.makedirs(images_dir, exist_ok=True)  # Ensure the directory exists
    output_path = os.path.join(images_dir, f"{patient_id}_landmarks_{int(time.time())}.png")
    save_image_with_landmarks(image_data, current_landmarks, output_path)

    # Encode the image to Base64
    _, buffer = cv2.imencode('.png', image_data)
    base64_image = base64.b64encode(buffer).decode('utf-8')

    # Return results
    return {
        "significant_change": significant_change,
        "change_value": change_value,
        "annotated_image": base64_image,
    }

def handle_image_from_api(base64_image, patient_id):
    """
    Handle an image sent via an API and return comparison results.
    :param base64_image: Base64-encoded image string.
    :param patient_id: The patient's unique identifier.
    :return: A dictionary with the comparison results.
    """
    # Decode the Base64 image
    image_bytes = base64.b64decode(base64_image)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Run comparison on the received image
    return run_comparison(image, patient_id)


@app.route('/api/upload', methods=['POST'])
def upload_image():
    data = request.json
    patient_name = data.get("name", "").strip()
    image_data = data.get("image", "")

    if not patient_name:
        return jsonify({"error": "Patient name is required"}), 400
    
    if not image_data:
        return jsonify({"error": "No image provided"}), 400
    
    try:
        response = handle_image_from_api(image_data, patient_name)
        if "error" in response:
            return jsonify({"error": response["error"]}), 400
        return jsonify({
            "message": "Processed successfully.",
            "significant_change": response["significant_change"],
            "change_value": response["change_value"],
            "annotated_image": response["annotated_image"]
        }), 200
    except Exception as e:
        print("Error processing image:", e)
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)