from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import io

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Disease classes
DISEASE_CLASSES = [
    "Healthy",
    "Bacterial Leaf Blight",
    "Leaf Spot",
    "Powdery Mildew",
    "Rust"
]

# Load the trained model (comment out if testing without model)
try:
    # Try to load the model from .h5 file
    model = tf.keras.models.load_model('model.h5')
    print("✅ Model loaded successfully.")
except Exception as e:
    # If loading fails, print the error and set model to None
    model = None
    print("⚠️ Warning: Model not loaded. Using simulated predictions.")
    print(f"Error: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


def predict_disease(image_bytes):
    img_array = preprocess_image(image_bytes)
    
    if model:
        predictions = model.predict(img_array)[0]
    else:
        predictions = np.random.rand(len(DISEASE_CLASSES))
    
    predictions = predictions / np.sum(predictions)
    return predictions


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file format. Please upload JPG or PNG.'})
    
    try:
        image_bytes = file.read()
        predictions = predict_disease(image_bytes)

        results = [
            {'disease': disease, 'probability': float(prob)}
            for disease, prob in zip(DISEASE_CLASSES, predictions)
        ]
        results.sort(key=lambda x: x['probability'], reverse=True)

        return jsonify({'success': True, 'predictions': results})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)

