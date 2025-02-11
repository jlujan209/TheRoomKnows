from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
from dotenv import load_dotenv
import os

app = Flask(__name__)
CORS(app)
load_dotenv()

conn = sqlite3.connect("patients.db", check_same_thread=False)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

API_KEY = os.getenv('API_KEY')

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
    
    data=request.get_json()
    cursor.execute('''SELECT * FROM patient_data WHERE patient_id=?''', (data.get('patient_id')))
    row = cursor.fetchone()

    return jsonify({"patient": dict(row)}), 200

@app.route('/patients/delete', methods=['DELETE'])
def delete_patient():
    api_key = request.headers.get('API-Key')
    if api_key != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
    data=request.get_json()
    cursor.execute('''DELETE FROM patient_data WHERE patient_id=?''', (data.get('patient_id')))
    conn.commit()
    return jsonify({"message": "Successful Deletion", "data": data}), 201

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
    id = data.get('patient_id')
    cursor.execute('''UPDATE patient_data 
                   SET patient_first_name=?, patient_last_name=?, patient_age=?, last_visit=? 
                   WHERE patient_id=?''', (first_name, last_name, age, last_visit, id))
    conn.commit()
    return jsonify({"message": "Updated Successful", "data": data}), 201

if __name__ == '__main__':
    app.run(debug=True, ssl_context=('./cert.pem', './key.pem'))