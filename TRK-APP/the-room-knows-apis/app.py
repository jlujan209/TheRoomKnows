import eventlet
eventlet.monkey_patch()

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity, JWTManager
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
import tensorflow as tf
import numpy as np
import sqlite3
import base64
import os
import cv2
import ssl

import eventlet.wsgi
import eventlet.greenio.base

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
ssl_context.load_cert_chain(certfile="./cert.pem", keyfile="./key.pem")

API_KEY = os.getenv('API_KEY')

# Dummy user db for dev
# TODO: Create user database with hashing
users = {
    "username": "password"
}

# Emotion Detection Model
ed_model = tf.keras.models.load_model("emotion_detection.keras")
emotion_labels = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]

@app.route('/login', methods=['POST'])
def login():
    api_key = request.headers.get('API-Key')
    if api_key != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
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
    api_key = request.headers.get('API-Key')
    if api_key != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    cursor.execute('''SELECT * FROM patient_data;''')
    patients = cursor.fetchall()
    patients_list = [dict(row) for row in patients]
    return jsonify({"patients": patients_list}), 200

@app.route('/patients/new', methods=['POST'])
def add_new_patient():
    api_key = request.headers.get('API_Key')
    if api_key != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
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
    api_key = request.headers.get('API-Key')
    if api_key != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
    id=request.args.get('patient_id')
    cursor.execute('''SELECT * FROM patient_data WHERE patient_id=?''', (id,))
    row = cursor.fetchone()

    return jsonify({"patient": dict(row)}), 200

@app.route('/patients/delete', methods=['DELETE'])
def delete_patient():
    api_key = request.headers.get('API-Key')
    if api_key != API_KEY:
        response = make_response(jsonify({"error": "Unauthorized"}), 401)
    else:
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
    api_key = request.headers.get('API-Key')
    if api_key != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
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
    print('Client Connected')
    emit("connection_response", {"message": "Connected to Websocket"})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected') 


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

if __name__ == '__main__':
    #app.run(debug=True, ssl_context=('./cert.pem', './key.pem'))
    # socketio.run(app, debug=True, host="0.0.0.0", port=5000, ssl_context=('./cert.pem', './key.pem'))
    #If not using with ssl use this instead: 
    # app.run(debug=True)
    listener = eventlet.listen(("0.0.0.0", 5000))
    secure_listener = eventlet.wrap_ssl(listener, certfile="./cert.pem", keyfile="./key.pem", server_side=True)

    print("Server running with SSL on port 5000...")
    print(f'PID : {os.getpid()}')

    # Use Eventlet's WSGI server to serve Flask with SSL
    eventlet.wsgi.server(secure_listener, app)