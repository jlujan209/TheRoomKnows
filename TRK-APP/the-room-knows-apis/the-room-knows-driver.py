from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os

app = Flask(__name__)
load_dotenv()

API_KEY = os.getenv('API_KEY')

@app.route('/patients', methods=['GET'])
def get_patients():
    api_key = request.headers.get('API-Key')
    print(f' Expected Key: {API_KEY}, Key Received: {api_key}')
    if api_key != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
    return jsonify({"message": "Success"})

if __name__ == '__main__':
    app.run(debug=True, ssl_context=('./cert.pem', './key.pem'))