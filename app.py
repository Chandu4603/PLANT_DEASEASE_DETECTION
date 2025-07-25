from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# Simulated disease classes (you can expand this list)
DISEASE_CLASSES = [
    "Healthy",
    "Bacterial Leaf Blight",
    "Leaf Spot",
    "Powdery Mildew",
    "Rust"
]

# Simulate a model prediction (replace this with actual model logic)
def predict_disease(image):
    # Convert image to format needed for prediction
    img = Image.open(io.BytesIO(image))
    img = img.resize((224, 224))  # Standard input size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Simulate prediction (random for demo)
    # In real application, you would use: prediction = model.predict(img_array)
    prediction = np.random.random(len(DISEASE_CLASSES))
    prediction = prediction / np.sum(prediction)  # Normalize to probabilities
    
    return prediction

# Create upload folder if it doesn't exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    try:
        # Read the image file
        img_bytes = file.read()
        
        # Get prediction
        predictions = predict_disease(img_bytes)
        
        # Format results
        results = [
            {'disease': disease, 'probability': float(prob)}
            for disease, prob in zip(DISEASE_CLASSES, predictions)
        ]
        
        # Sort by probability
        results.sort(key=lambda x: x['probability'], reverse=True)
        
        return jsonify({
            'success': True,
            'predictions': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)