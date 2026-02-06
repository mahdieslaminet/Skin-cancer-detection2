# Skin Cancer Detection API

A Flask-based web application that uses a deep learning model to detect skin cancer from images. This implementation is based on research in deep learning for skin cancer detection using convolutional neural networks (CNNs).

## Background

Skin cancer is one of the most common types of cancer worldwide, with its frequency increasing globally. The main subtypes include:
- **Basal cell carcinoma** - Most common form
- **Squamous cell carcinoma** - Second most common
- **Melanoma** - Most aggressive and responsible for most skin cancer-related deaths

Early detection and accurate diagnosis are crucial for effective treatment. Deep learning methods, particularly CNNs, have shown promising results in accurately and swiftly identifying skin cancer from dermoscopic images.

This project implements a CNN-based classification system inspired by research on skin lesion classification using the ISIC (International Skin Imaging Collaboration) dataset.

## Features

- 🖼️ Upload skin lesion images via a simple web UI
- 🤖 AI-powered classification using a trained CNN model
- 📊 Detailed prediction results with confidence scores
- 🎨 Modern, responsive user interface
- 🔬 Classification of 9 different skin lesion types

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train and Save the Model

Open `skin_cancer_model.ipynb` in Jupyter and run all cells. The notebook will:
- Download the ISIC (International Skin Imaging Collaboration) skin cancer dataset
- Preprocess images (resize, normalize, augment)
- Train a custom CNN model with the following architecture:
  - Two convolutional layers (32 and 64 filters)
  - Max pooling layers
  - Dense layers with dropout for regularization
  - Output layer with 9 classes
- Save the model to `models/skin_cancer_model.h5`

Alternatively, if you already have a trained model, place it at `models/skin_cancer_model.h5`.

**Note:** The model uses a lightweight CNN architecture optimized for speed while maintaining reasonable accuracy. For production use, consider fine-tuning with transfer learning models like ResNet50, InceptionV3, or Inception ResNet for higher accuracy (as demonstrated in research achieving 83-86% accuracy).

### 3. Run the Flask Application

```bash
python app.py
```

The application will start on `http://localhost:5001`

### 4. Use the Web Interface

1. Open your browser and go to `http://localhost:5001`
2. Click or drag and drop a skin lesion image
3. Click "Analyze Image" to get predictions
4. View the results showing whether cancer is detected and confidence scores

## API Endpoints

### `GET /`
Serves the main web UI.

### `POST /predict`
Upload an image for prediction.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: `file` (image file)

**Response:**
```json
{
  "success": true,
  "prediction": {
    "class_name": "Melanoma",
    "is_cancerous": true,
    "confidence": 0.85,
    "all_predictions": {
      "Melanoma": 0.85,
      "Benign keratosis": 0.10,
      ...
    }
  }
}
```

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## Project Structure

```
aiMedicine/
├── app.py                      # Flask backend application
├── templates/
│   └── index.html             # Web UI
├── models/
│   └── skin_cancer_model.h5  # Trained model (generated after training)
├── uploads/                   # Temporary upload directory
├── skin_cancer_model.ipynb    # Jupyter notebook for training
├── skin_cancer_model.py       # Python script version
└── requirements.txt           # Python dependencies
```

## Model Information

### Architecture

The model uses a custom Convolutional Neural Network (CNN) architecture:
- **Input**: 128x128x3 RGB images
- **Convolutional Layers**: 
  - Conv2D(32 filters, 3x3) + ReLU + MaxPooling
  - Conv2D(64 filters, 3x3) + ReLU + MaxPooling
- **Dense Layers**: 
  - Flatten + Dense(64) + ReLU + Dropout(0.5)
  - Dense(9) + Softmax (output layer)

### Classification Categories

The model classifies skin lesions into 9 categories based on the ISIC dataset:
- **Actinic keratosis** - Precancerous growth
- **Basal cell carcinoma** ⚠️ (cancerous)
- **Benign keratosis** - Non-cancerous
- **Dermatofibroma** - Benign skin lesion
- **Melanoma** ⚠️ (cancerous - most dangerous)
- **Melanocytic nevi** - Benign moles
- **Squamous cell carcinoma** ⚠️ (cancerous)
- **Vascular lesion** - Blood vessel abnormalities
- **Unknown** - Unclassified lesions

### Dataset

This project uses the ISIC (International Skin Imaging Collaboration) dataset, which contains thousands of dermoscopic images of skin lesions. The dataset is widely used in research for skin cancer detection and classification.

### Performance

Based on research in deep learning for skin cancer detection:
- Custom CNN models typically achieve **80-85% accuracy**
- Transfer learning models (ResNet50, InceptionV3) can achieve **83-86% accuracy**
- Model performance depends on:
  - Dataset size and quality
  - Image preprocessing and augmentation
  - Model architecture and hyperparameters
  - Training duration and techniques

**Note:** The actual accuracy of your trained model may vary based on training parameters and dataset characteristics.

## Important Notes

⚠️ **This application is for educational and research purposes only. Always consult a medical professional for accurate diagnosis.**

- The model accuracy may vary and should not be used as a substitute for professional medical advice
- Skin cancer diagnosis requires clinical examination by trained dermatologists
- Dermoscopic images should be interpreted by medical professionals
- This tool is intended to assist, not replace, medical diagnosis

## Research Reference

This implementation is based on research in deep learning for skin cancer detection. For more information, refer to:

**Gouda, W., et al.** (2022). "Detection of Skin Cancer Based on Skin Lesion Images Using Deep Learning." *Healthcare (Basel)*, 10(7), 1183. doi: 10.3390/healthcare10071183

Available at: https://pmc.ncbi.nlm.nih.gov/articles/PMC9324455/

The research demonstrates the effectiveness of CNN and transfer learning approaches (ResNet50, InceptionV3, Inception ResNet) for skin lesion classification using the ISIC dataset, achieving accuracy rates of 83-86%.

## Troubleshooting

### Model not found error
- Make sure you've trained and saved the model using the notebook
- Check that `models/skin_cancer_model.h5` exists

### Import errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Make sure you're using the correct Python version (3.8+)

### Port already in use
- The default port is 5001 to avoid conflicts with macOS AirPlay Receiver (which uses port 5000)
- To change the port, edit `app.py`: `app.run(debug=True, host='0.0.0.0', port=YOUR_PORT)`

