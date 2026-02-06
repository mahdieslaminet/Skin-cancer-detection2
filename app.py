import os
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)

# Global variable to store the model
model = None

# Class names (9 categories from the ISIC dataset)
CLASS_NAMES = [
    'Actinic keratosis',
    'Basal cell carcinoma',
    'Benign keratosis',
    'Dermatofibroma',
    'Melanoma',
    'Melanocytic nevi',
    'Squamous cell carcinoma',
    'Vascular lesion',
    'Unknown'
]

# Cancerous classes (indices that indicate cancer)
CANCEROUS_CLASSES = [1, 4, 6]  # Basal cell carcinoma, Melanoma, Squamous cell carcinoma

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_model_if_exists():
    """Load the trained model if it exists"""
    global model
    model_path = 'models/skin_cancer_model.h5'
    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
            print(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    else:
        print(f"Model not found at {model_path}")
        print("Please train the model first using the notebook and save it.")
        return False

def preprocess_image(img):
    """Preprocess image for model prediction"""
    # Resize to model input size (128, 128)
    img = img.resize((128, 128))
    # Convert to array and normalize
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_image(img):
    """Predict skin cancer from image"""
    if model is None:
        return None, "Model not loaded. Please train and save the model first."
    
    try:
        # Preprocess image
        img_array = preprocess_image(img)
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get class name
        class_name = CLASS_NAMES[predicted_class_idx]
        
        # Determine if cancerous
        is_cancerous = predicted_class_idx in CANCEROUS_CLASSES
        
        # Get all class probabilities
        all_predictions = {
            CLASS_NAMES[i]: float(predictions[0][i]) 
            for i in range(len(CLASS_NAMES))
        }
        
        return {
            'class_name': class_name,
            'is_cancerous': is_cancerous,
            'confidence': confidence,
            'all_predictions': all_predictions
        }, None
    except Exception as e:
        return None, str(e)

@app.route('/')
def index():
    """Serve the main UI page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for image prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload an image (png, jpg, jpeg, gif, bmp)'}), 400
    
    try:
        # Read image from file
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Make prediction
        result, error = predict_image(img)
        
        if error:
            return jsonify({'error': error}), 500
        
        return jsonify({
            'success': True,
            'prediction': result
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    model_loaded = model is not None
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded
    }), 200

if __name__ == '__main__':
    # Try to load model on startup
    load_model_if_exists()
    
    # Run the app
    print("Starting Flask server...")
    print("Visit http://localhost:5001 to use the UI")
    if model is None:
        print("WARNING: Model not loaded. Please train and save the model first.")
    app.run(debug=True, host='0.0.0.0', port=5001)

