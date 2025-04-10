import eventlet
import sys
import threading
import requests
eventlet.monkey_patch()

from flask import Flask, request, jsonify, make_response, send_file
from flask_cors import CORS
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity, JWTManager
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
import tensorflow
import numpy as np
import sqlite3
import base64
import os
import cv2
import ssl
import tempfile
from datetime import datetime
import json
import pickle
import mediapipe as mp
import time

import eventlet.wsgi
import eventlet.greenio.base

import sounddevice as sd
import soundfile as sf
import openai
import whisper
from group_by_qa import query_openai, perform_sentiment_analysis, perform_frequency_analysis
import matplotlib.pyplot as plt
import pandas as pd
import re
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle, Image
from reportlab.lib import colors

from gait_analyzer import analyze_gait

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Flask app with CORS enabled
app = Flask(__name__)
CORS(app)
load_dotenv()

# For SQLite Connection:
conn = sqlite3.connect("patients.db", check_same_thread=False)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# For Login token:
jwt = JWTManager(app)
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_KEY')

# For Websocket
socketio = SocketIO(app, async_mode="eventlet") 

# SSL
ssl._create_default_https_context = ssl._create_unverified_context
ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

API_KEY = os.getenv('API_KEY')

# Dummy user db for dev
# TODO: Create user database with hashing
users = {
    "username": "password"
}

# stuff skyler added for emotion and audio stuff
current_audio_file = None
current_file_lock = threading.Lock()
transcripts = []
emotions = {
        "Angry": 0,
        "Happy": 0,
        "Neutral": 0,
        "Sad": 0,
        "Surprise": 0
    }
OPENAI_WHISPER_ENDPOINT = 'https://api.openai.com/v1/audio/transcriptions'
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY
TEMPDIR = os.getenv('TEMP_DIR', '/tmp')
model = whisper.load_model('base')

# Emotion Detection Model
print(tensorflow.__version__)
ed_model = tensorflow.keras.models.load_model("emotion_detection.keras")
emotion_labels = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]

@app.route('/')
def index():
    return jsonify({"message": "Welcome to The Room Knows API"}), 200

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    # Check if user exists and password matches
    if username in users and users[username] == password:
        token = create_access_token(identity=username)
        return jsonify(access_token=token), 200

    return jsonify({"error": "Invalid credentials"}), 401


@app.route('/patients/all', methods=['GET'])
def get_all_patients(): 
    cursor.execute('''SELECT * FROM patient_data;''')
    patients = cursor.fetchall()
    patients_list = [dict(row) for row in patients]
    return jsonify({"patients": patients_list}), 200

@app.route('/patients/new', methods=['POST'])
def add_new_patient():
    
    data=request.get_json()
    first_name = data.get("patient_first_name")
    last_name = data.get("patient_last_name")
    age = data.get("patient_age")
    last_visit = data.get("last_visit")

    cursor.execute('''INSERT OR REPLACE INTO patient_data (patient_first_name,  patient_last_name, patient_age, last_visit)
                   VALUES (?,?,?,?)''', (first_name, last_name, age, last_visit))
    conn.commit()

    return jsonify({
        "message" : "Patient added",
        "data": data
    }), 201
    
@app.route('/patients/search', methods=['GET'])
def get_patient():
    
    id=request.args.get('patient_id')
    cursor.execute('''SELECT * FROM patient_data WHERE patient_id=?''', (id,))
    row = cursor.fetchone()

    return jsonify({"patient": dict(row)}), 200

@app.route('/patients/delete', methods=['DELETE'])
def delete_patient():
    id = request.args.get('patient_id')
    cursor.execute('''DELETE FROM patient_data WHERE patient_id=?''', (id,))
    conn.commit()
    response = make_response(jsonify({"message": "Successful Deletion"}), 201)
    
    # Set headers
    response.headers['Content-Type'] = 'application/json'
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST,PATCH,OPTIONS'
    
    return response

@app.route('/patients/edit', methods=['PUT'])
def edit_patient():
    
    data=request.get_json()
    first_name = data.get('patient_first_name')
    last_name = data.get('patient_last_name')
    age = data.get('patient_age')
    last_visit = data.get('last_visit')
    id = request.args.get('patient_id')
    cursor.execute('''UPDATE patient_data 
                   SET patient_first_name=?, patient_last_name=?, patient_age=?, last_visit=? 
                   WHERE patient_id=?''', (first_name, last_name, age, last_visit, id))
    conn.commit()
    return jsonify({"message": "Updated Successful", "data": data}), 201

# Web Sockets Routes to handle real-time connections
@socketio.on('connect')
def handle_connect():
    start_session()
    print('Session started')
    emit("connection_response", {"message": "Speech Analysis Session Started"})

@socketio.on('disconnect')
def handle_disconnect():
    final_transcripts, emotions = stop_session()
    with open("transcripts.txt", "w") as f:
        for transcript in final_transcripts:
            f.write(transcript + "\n")
    with open("emotions.json", "w") as f:
        json.dump(emotions, f)
    print("Client Disconnected, Speech Analysis Session has ended")


@app.route("/analysis/emotion-detection", methods=["POST"])
def predict_emotion():
    data = request.json
    image_data = data.get("image", "")
    if not image_data:
        return jsonify({"error": "No Image Provided"}), 400
    try: 
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (224, 224))
        image = image / 255.0
        image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0) 

        prediction = ed_model.predict(image)
        emotion = emotion_labels[np.argmax(prediction)]
    except:
        return jsonify({"error": "Failed to predict emotion"})

    return jsonify({"emotion": emotion, "confidence": float(np.max(prediction))})


def record_audio():
    global current_audio_file

    device = sd.query_devices(kind='input')
    mic = device['name']

    while session_active:
        with tempfile.NamedTemporaryFile(dir='./tmp', delete=False, suffix=".wav") as tmpfile:
            filename = tmpfile.name

        current_audio_file = sf.SoundFile(filename, mode='w', samplerate=44100, channels=1)

        def callback(indata, frames, time_info, status):
            if not session_active:
                raise sd.CallbackStop()
            with current_file_lock:
                current_audio_file.write(indata)

        with sd.InputStream(samplerate=44100, channels=1, device=mic, callback=callback):
            print(f"[AUDIO] Recording to {filename}")
            while session_active:
                eventlet.sleep(0.1)

        print("[AUDIO] Recording stopped.")

def transcribe_audio_file(filepath):
    print(f"[TRANSCRIBE] Sending {filepath} to OpenAI...")
    try:
        if not os.path.exists(filepath):
            print(f"[TRANSCRIBE] File does not exist: {filepath}")
            return
        print(f"[TRANSCRIBE] File exists: {filepath}")
        if not os.path.isfile(filepath):
            print(f"[TRANSCRIBE] Filepath is invalid or not a file: {filepath}")
            return
            
        response = model.transcribe(filepath)
        # Save the response to a file for debugging
        with open("response.json", "w") as f:
            json.dump(response, f)
        transcription = response['text']
        print(f"[TRANSCRIBE] Result: {transcription}")
        transcripts.append(transcription)
    except requests.exceptions.RequestException as e:
        print(f"[TRANSCRIBE] HTTP request failed: {e}")
    except Exception as e:
        print(f"[TRANSCRIBE] Failed to transcribe: {e}")
    finally:
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"[TRANSCRIBE] Failed to remove file: {e}")

def rotate_and_transcribe():
    global current_audio_file

    while session_active:
        eventlet.sleep(240)  # Wait 4 minutes
        if not session_active:
            break

        with current_file_lock:
            old_file = current_audio_file.name
            current_audio_file.close()

        # Spawn transcription task
        eventlet.spawn(transcribe_audio_file, old_file)

        # Create new file for continued recording
        with tempfile.NamedTemporaryFile(dir='./tmp', delete=False, suffix=".wav") as tmpfile:
            new_filename = tmpfile.name

        with current_file_lock:
            current_audio_file = sf.SoundFile(new_filename, mode='w', samplerate=44100, channels=1)

def emotion_analysis():
    print("[EMOTION] Starting emotion analysis...")
    cur_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs("photos", exist_ok=True)
    os.makedirs(f"photos/{cur_time}", exist_ok=True)
    camera = cv2.VideoCapture(0)  # Open the camera
    if not camera.isOpened():
        print("Error: Could not access the camera.")
        return
    while session_active:
        print(emotions)
        ret, frame = camera.read()
        if not ret:
            print("Failed to capture image.")
            continue
        try: 
            image = cv2.resize(frame, (224, 224))
            image = image / 255.0
            image = np.expand_dims(image, axis=0)

            prediction = ed_model.predict(image)
            emotion = emotion_labels[np.argmax(prediction)]
            print(f"Emotion detected: {emotion}")
            # keep counds
            emotions[emotion] += 1
            # TODO: save the emotion to the db or something?

        except Exception as e:
            print(f"[EMOTION] Failed to predict emotion: {e}")

        eventlet.sleep(5)  # Sleep for a second before capturing the next image

def start_session():
    global session_active, transcripts
    session_active = True
    transcripts = []
    eventlet.spawn(record_audio)
    eventlet.spawn(rotate_and_transcribe)
    eventlet.spawn(emotion_analysis)

def stop_session():
    global session_active
    session_active = False

    # Finalize last chunk
    with current_file_lock:
        final_file = current_audio_file.name
        current_audio_file.close()

    transcribe_audio_file(final_file)  # Send last partial chunk
    all_text = " ".join(transcripts)
    print(f"[TRANSCRIBE] All text: {all_text}")
    # run frequency analysis on the text
    frequency_analysis = perform_frequency_analysis(all_text)
    # write the frequency analysis to a file
    with open("frequency_analysis.json", "w") as f:
        json.dump(frequency_analysis, f)
    # get qa pairs for sentiment analysis
    qas = query_openai(all_text)
    # perform sentiment analysis on the text
    sentiment_analyzer = perform_sentiment_analysis(qas)
    # write the sentiment analysis to a file
    with open("sentiment_analysis.json", "w") as f:
        json.dump(sentiment_analyzer, f)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(f"emotions_{timestamp}.json", "w") as f:
        json.dump(emotions, f)

    return all_text, emotions

@app.route('/analysis/emotion-detection/save-results', methods=['POST'])
def save_results():
    data = request.get_json()
    results = data.get('results')
    patient_name = data.get('patient_name')
    try:
        cursor.execute('INSERT INTO patient_analysis (patient_id, analysis_type, value) VALUES (?,?,?)', (patient_name, 'emotion', json.dumps(results),))
        conn.commit()
        return jsonify({"message": "successfully saved results for emotion detection."}), 201
    except Exception as e:
        return jsonify({"error": e}), 500




### Facial Mapping -------------------------------------------------------------------------------
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




def calculate_asymmetry(landmarks):
    """
    Calculate asymmetry by comparing left and right landmarks.
    :param landmarks: A NumPy array of facial landmarks.
    :return: A dictionary with asymmetry metrics.
    """
    asymmetries = {}

    # Define landmark pairs for left and right regions (MediaPipe IDs)
    pairs = {
        "eyes": [33, 263],  # Left eye, right eye
        "cheeks": [234, 454],  # Left cheek, right cheek
        "mouth": [61, 291],  # Left corner of mouth, right corner of mouth
    }

    for region, (left_idx, right_idx) in pairs.items():
        left = landmarks[left_idx]
        right = landmarks[right_idx]
        # Calculate Euclidean distance between corresponding landmarks
        asymmetries[region] = np.linalg.norm(left - right)

    return asymmetries


def check_asymmetry_changes(current_asymmetry, previous_asymmetry, thresholds):
    changes = {}
    for region, current_value in current_asymmetry.items():
        previous_value = previous_asymmetry.get(region, 0.0)
        difference = abs(current_value - previous_value)
        if difference > thresholds.get(region, 0.1):  # Default threshold is 0.1
            changes[region] = difference
    print(f"Current asymmetry: {current_asymmetry}")
    print(f"Previous asymmetry: {previous_asymmetry}")
    print(f"Detected changes: {changes}")
    return changes

@app.route('/load-test-data', methods=['GET'])
def load_test_data():
    # drop all rows from patient_analysis
    cursor.execute('''
    DELETE FROM patient_analysis
    ''')
    cursor.execute('''
    INSERT INTO patient_analysis (patient_id, analysis_type, value, created_date)
    VALUES (?, ?, ?, ?)
    ''',(22, "emotion", json.dumps({"happy": 19, "sad": 10, "angry": 5, "surprise": 2, "neutral": 20}), "2025-01-01"))
    conn.commit()
    cursor.execute('''
    INSERT INTO patient_analysis (patient_id, analysis_type, value, created_date)
    VALUES (?, ?, ?, ?)
    ''',(22, "emotion", json.dumps({"happy": 10, "sad": 13, "angry": 7, "surprise": 3, "neutral": 21}), "2025-03-22"))
    conn.commit()
    cursor.execute('''
    INSERT INTO patient_analysis (patient_id, analysis_type, value, created_date)
    VALUES (?, ?, ?, ?)
    ''', (22, "frequency", json.dumps({"patient_symptoms": [
            {"symptom": "chest pain", "count": 5},
            {"symptom": "sweating", "count": 3},
            {"symptom": "nausea", "count": 1}
        ]}),"2025-01-01"))
    conn.commit()
    cursor.execute('''
    INSERT INTO patient_analysis (patient_id, analysis_type, value, created_date)
    VALUES (?, ?, ?, ?)
    ''', (22, "frequency", json.dumps({"patient_symptoms": [
            {"symptom": "chest pain", "count": 3},
            {"symptom": "sweating", "count": 1}
        ]}),"2025-03-22"))
    conn.commit()
    cursor.execute('''
    INSERT INTO patient_analysis (patient_id, analysis_type, value, created_date)
    VALUES (?, ?, ?, ?)   
    ''',(22, 'sentiment', json.dumps({
        'positive': 10,
        'negative': 11,
        'neutral': 20,
    }), "2025-03-22"))
    conn.commit()
    cursor.execute('''
    INSERT INTO patient_analysis (patient_id, analysis_type, value, created_date)
    VALUES (?, ?, ?, ?)   
    ''',(22, 'sentiment', json.dumps({
        'positive': 20,
        'negative': 8,
        'neutral': 20,
    }), "2025-01-01"))
    conn.commit()
    return jsonify({
        "message" : "canned data added",
        "data": "data"
    }), 201
    
def capture_image():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam")
        return None
    
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print('Failed to read frame from webcam')
        return None
    
    return frame

def run_comparison(image_data, patient_id):
    """
    Compare the facial asymmetry of the given image with the previously saved asymmetry.
    Save the image with landmarks annotated and return comparison results.
    """
    # Extract landmarks from the current image
    current_landmarks = extract_facial_landmarks(image_data)
    if len(current_landmarks) == 0:
        return {"error": "No face detected in the image."}

    # Calculate current asymmetry metrics
    current_asymmetry = calculate_asymmetry(current_landmarks)

    # Fetch previous asymmetry metrics from the database
    previous_landmarks = fetch_landmarks(patient_id)
    significant_change = False
    if previous_landmarks is None:
        previous_asymmetry = {}
        change_value = 0.0
    else:
        previous_asymmetry = calculate_asymmetry(previous_landmarks)
        thresholds = {"eyes": 0.025, "cheeks": 0.025, "mouth": 0.025}
        significant_changes = check_asymmetry_changes(current_asymmetry, previous_asymmetry, thresholds)
        significant_change = len(significant_changes) > 0
        change_value = max(significant_changes.values()) if significant_changes else 0.0

    # Save the current landmarks for future comparisons
    save_landmarks(patient_id, current_landmarks)

    # Save the annotated image
    images_dir = "./images"
    os.makedirs(images_dir, exist_ok=True)
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

@app.route('/analysis/facial-mapping', methods=['POST'])
def upload_image():
    start_time = time.perf_counter()

    data = request.json
    patient_name = data.get("name", "").strip()
    image_data = data.get("image", "")

    if not patient_name:
        return jsonify({"error": "Patient name is required"}), 400

    if not image_data:
        return jsonify({"error": "No image provided"}), 400

    try:
        response = handle_image_from_api(image_data, patient_name)
        
        end_time = time.perf_counter()
        execution_time = end_time - start_time

        print(f"Execution time: {execution_time} seconds")
        if "error" in response:
            return jsonify({"error": response["error"]}), 400
        return jsonify({
            "message": "Processed successfully.",
            "significant_change": response["significant_change"],
            "change_value": response["change_value"],
            "annotated_image": response["annotated_image"],
        }), 200
    except Exception as e:
        print("Error processing image:", e)
        return jsonify({"error": str(e)}), 500

@app.route('/generate-report/<patient_id>', methods=['GET'])
def generate_report(patient_id: str):
    start_time = time.time()
    # get emotion data
    cursor.execute('''
    SELECT * FROM patient_analysis WHERE patient_id = ?
    AND analysis_type = 'emotion'
    ''', (patient_id,))
    rows = cursor.fetchall()
    # get the two most recent rows by date
    rows = sorted(rows, key=lambda x: x['created_date'], reverse=True)[:2]
    # check for significant change
    if len(rows) < 2:
        emotion_conclusion = "only one visit was recorded, no significant change detected"
    else:
        rows[0] = json.loads(rows[0]['value'])
        rows[1] = json.loads(rows[1]['value'])
        emotion_conclusion = "significant change detected in emotions: "
        change_detected_in = []
        if abs(rows[0]['neutral'] - rows[1]['neutral']) > 10:
            change_detected_in.append("neutral")
        if abs(rows[0]['happy'] - rows[1]['happy']) > 10:
            change_detected_in.append("happy")
        if abs(rows[0]['sad'] - rows[1]['sad']) > 10:
            change_detected_in.append("sad")
        if abs(rows[0]['angry'] - rows[1]['angry']) > 10:
            change_detected_in.append("angry")
        if abs(rows[0]['surprise'] - rows[1]['surprise']) > 10:
            change_detected_in.append("surprise")
        if len(change_detected_in) == 0:
            emotion_conclusion = "no significant change detected"
        else:
            emotion_conclusion += ", ".join(change_detected_in)
    
    # create a plot of most recent visit
    plt.figure(figsize=(10, 6))
    plt.bar(rows[0].keys(), rows[0].values())
    plt.title(f"Patient {patient_id} Emotions")
    plt.xlabel("Emotions")
    plt.ylabel("Count")
    plt.savefig(f"graphs/{patient_id}_emotion_analysis.png")
    plt.close()

    # get the most recent frequency analysis
    cursor.execute('''
    SELECT * FROM patient_analysis WHERE patient_id = ?
    AND analysis_type = 'frequency'
    ''', (patient_id,))

    # get the most recent row by date
    rows = cursor.fetchall()
    rows = sorted(rows, key=lambda x: x['created_date'], reverse=True)
    if len(rows) == 0:
        frequency_conclusion = "no frequency analysis was recorded"
    else:
        new_rows = []
        freq_data = json.loads(rows[0]['value'])
        print("Frequency Analysis Data:")
        print(freq_data)
        print(rows)
        for row in rows:
            print(row.keys())
            new_rows.append(json.loads(row['value']))
            new_rows[-1]['date'] = row['created_date']

        # Convert to DataFrame
        data_rows = []
        # Flatten the data into a list of dictionaries
        for entry in new_rows:
            date = entry['date']
            for symptom in entry['patient_symptoms']:
                data_rows.append({
                    'date': date,
                    'symptom': symptom['symptom'],
                    'count': symptom['count']
                })

        df = pd.DataFrame(data_rows)

        # Convert 'date' column to datetime
        df['date'] = pd.to_datetime(df['date'])

        # Pivot the DataFrame to have a column for each symptom
        df_pivot = df.pivot_table(index='date', columns='symptom', values='count', aggfunc='sum').fillna(0)

        # Plotting
        plt.figure(figsize=(10, 6))

        # Plot each symptom's count over time
        for symptom in df_pivot.columns:
            plt.plot(df_pivot.index, df_pivot[symptom], label=symptom)

        plt.title('Symptom Complaints Over Time')
        plt.xlabel('Date')
        plt.ylabel('Count')
        plt.legend(title='Symptoms')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"graphs/{patient_id}_frequency_analysis.png")
        plt.close()

        frequency_conclusion = "done"
    
    # analyze the sentiment analysis
    cursor.execute('''
    SELECT * FROM patient_analysis WHERE patient_id = ?
    AND analysis_type = 'sentiment'
    ''', (patient_id,))
    rows = cursor.fetchall()
    # get the most recent row by date
    rows = sorted(rows, key=lambda x: x['created_date'], reverse=True)
    if len(rows) == 0:
        sentiment_conclusion = "no sentiment analysis was recorded"
    else:
        rows[0] = json.loads(rows[0]['value'])
        sentiment_conclusion = "sentiment analysis: "
        if rows[0]['positive'] > rows[0]['negative']:
            sentiment_conclusion += "positive"
        elif rows[0]['positive'] < rows[0]['negative']:
            sentiment_conclusion += "negative"
        else:
            sentiment_conclusion += "neutral"
    
    # create a bar chart with sentiment on x ashix and count on y axis
    plt.figure(figsize=(10, 6))
    plt.bar(rows[0].keys(), rows[0].values())
    plt.title(f"Patient {patient_id} Sentiment Analysis")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.savefig(f"graphs/{patient_id}_sentiment_analysis.png")
    plt.close()

    generate_pdf_report(f"graphs/{patient_id}_frequency_analysis.png", freq_data, f"graphs/{patient_id}_sentiment_analysis.png", f"graphs/{patient_id}_emotion_analysis.png")

    end_time = time.time()
    elapsed_time = end_time - start_time
    return jsonify({
        "message": f"Report generated successfully in {elapsed_time:.2f} seconds",
        "emotion_analysis": {
            "conclusion": emotion_conclusion,
            "image": f"{patient_id}_emotion_analysis.png"
        },
        "frequency_analysis": {
            "conclusion": frequency_conclusion,
            "image": f"{patient_id}_frequency_analysis.png"
        }
    }), 200

def generate_pdf_report(freq_analysis_img, symptoms, sentiment_img, emotion_analysis_img):
    c = canvas.Canvas("report.pdf", pagesize=letter)
    width, height = letter
    cur_y = height - 50

    # Add title
    c.setFont("Helvetica", 20)
    c.drawString(100, cur_y, "Doctor Visit Analysis Report")
    c.setFont("Helvetica", 18)
    cur_y -= 20
    c.drawString(100, cur_y, "SUBJECTIVE")
    c.setFont("Helvetica", 16)
    cur_y -= 20
    c.drawString(100, cur_y, "Emotion Analysis Output")
    cur_y -= 320
    c.drawImage(emotion_analysis_img, 100, cur_y, width=400, height=300)

    cur_y -= 20
    c.setFont("Helvetica", 18)
    c.drawString(100, cur_y, "OBJECTIVE")
    c.setFont("Helvetica", 16)
    cur_y -= 20
    # Add sentiment plot
    c.drawString(100, cur_y, "Sentiment Analysis Output")
    cur_y -= 320
    c.drawImage(sentiment_img, 100, cur_y, width=400, height=300)
    cur_y -= 50
    c.showPage()
    c.setFont("Helvetica", 16)
    cur_y = height - 50
    c.drawString(100, cur_y, "Chief Complaint Counts")
    # Create a table for the counts
    data = [["Symptom", "Count"]]
    print(symptoms)
    print(symptoms['patient_symptoms'])
    for symptom in symptoms['patient_symptoms']:
        data.append([symptom["symptom"], symptom["count"]])
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    table.wrapOn(c, width, height)
    print(table.__dict__)
    cur_y -= (table._height + 10)
    table.drawOn(c, 100, cur_y)
    # Add the frequency analysis plot
    cur_y -= 20
    c.drawString(100, cur_y, "Frequency Analysis Output")
    cur_y -= 320
    c.drawImage(freq_analysis_img, 100, cur_y, width=400, height=300)
    cur_y -= 50
    c.setFont("Helvetica", 18)
    c.drawString(100, cur_y, "ASSESSMENT")
    cur_y -= 50
    c.drawString(100, cur_y, "PLAN")


    # Save the PDF
    c.save()

# Motion Analysis ----------------------------------------------------------------------------------
@app.route("/motion-analysis/upload-video", methods=["POST"])
def motion_analysis():
    
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the video file to the specified folder
    save_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
    video_file.save(save_path)

    # Check file size and OpenCV readability
    file_size = os.path.getsize(save_path)
    if file_size == 0:
        return jsonify({"error": "Uploaded file is empty"}), 400

    cap = cv2.VideoCapture(save_path)
    if not cap.isOpened():
        return jsonify({"error": "Failed to open video file with OpenCV"}), 500
    cap.release()

    # Proceed with gait analysis
    result, output_path = analyze_gait(save_path, output_dir=UPLOAD_FOLDER)

    return jsonify({
        "result": result,
        "annotated_video_url": f"/download/{os.path.basename(output_path)}"
    }), 203

@app.route('/motion-analysis/save-results', methods=['POST'])
def save_motion_analysis_results():
    data = request.get_json()
    patient_name = data.get('patient_name')
    result = data.get('result')

    if not patient_name or not result:
        return jsonify({'error': 'patient name and analysis result required'}), 400

    try:
        cursor.execute('INSERT INTO patient_analysis (patient_id, analysis_type, value) VALUES (?, ?, ?)', (patient_name, 'motion', result,))
        conn.commit()
    except Exception as e:
        print('Error uploading motion analysis results into db: ', e)
        return jsonify({"error": "Could not save results to the database."}), 500
    
    return jsonify({"message": "Result saved successfully."}), 200
    
@app.route('/facial-analysis/save-results', methods=['POST'])
def save_facial_feature_analysis_results():
    data = request.get_json()
    patient_name = data.get('patient_name')
    result = data.get('result')

    if not patient_name or not result:
        return jsonify({'error': 'patient name and analysis result required'}), 400

    try:
        cursor.execute('INSERT INTO patient_analysis (patient_id, analysis_type, value) VALUES (?, ?, ?)', (patient_name, 'facial', result,))
        conn.commit()
    except Exception as e:
        print('Error uploading facial feature analysis results into db: ', e)
        return jsonify({"error": "Could not save results to the database."}), 500
    
    return jsonify({"message": "Result saved successfully."}), 200

if __name__ == '__main__':
    #app.run(debug=True, ssl_context=('./cert.pem', './key.pem'))
    # socketio.run(app, debug=True, host="0.0.0.0", port=5000, ssl_context=('./cert.pem', './key.pem'))
    #If not using with ssl use this instead: 
    # app.run(debug=True)
    listener = eventlet.listen(("0.0.0.0", 5000))

    print("Server running with SSL on port 5000...")
    print(f'PID : {os.getpid()}')

    os.makedirs("tmp", exist_ok=True)

    # Use Eventlet's WSGI server to serve Flask with SSL
    eventlet.wsgi.server(listener, app)