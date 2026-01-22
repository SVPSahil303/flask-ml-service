from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np
import joblib
import io
from PIL import Image
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

model = joblib.load("model/teach2")  # Your trained model path

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data['image']

        # Remove the "data:image/png;base64," prefix
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('L')  # Convert to grayscale

        # Optional: Save image for debugging

        # Resize
        image = image.resize((28, 28), Image.Resampling.LANCZOS)

        # Convert to NumPy array
        image_array = np.array(image)

        # Apply threshold
        _, im_th = cv2.threshold(image_array, 100, 255, cv2.THRESH_BINARY)

        # Flatten and preprocess
        X = [1 if pixel > 100 else 0 for row in im_th for pixel in row]

        prediction = model.predict([X])[0]

        if prediction == 21:
            prediction = "null"

        return jsonify({'result': str(prediction)})
    except Exception as e:
        print("Prediction error:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
