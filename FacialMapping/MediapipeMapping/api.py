from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app)

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
        image_data = image_data.split(",")[1]
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        filename = f"{patient_name.replace(' ', '_')}_image.png"
        image.save(filename)

        return jsonify({"message": f"Image received and saved as {filename}."}),200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__=='__main__':
    app.run(debug=True)