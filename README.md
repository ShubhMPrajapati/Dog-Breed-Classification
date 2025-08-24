# 🐕 Dog Vision – AI-Powered Dog Breed Classifier

**Dog Vision** is an end-to-end deep learning project that classifies over **120 dog breeds** from images using **TensorFlow**, **MobileNetV2**, and **Transfer Learning**. The model is deployed as a **Flask web application**, allowing users to upload images and get **real-time predictions**.

---

## Features

- Predict over 120 dog breeds from a single image.
- Real-time predictions via **Flask web app**.
- Confidence scores and top-10 predictions visualization.
- Mobile-friendly and easy-to-use interface.
- Saved model in `.keras` and `.h5` formats for reproducibility.

---

## Tech Stack

- **Programming Languages:** Python  
- **Deep Learning:** TensorFlow, Keras, TensorFlow Hub, NumPy, Pandas  
- **Web Development:** Flask, HTML, CSS  
- **Visualization:** Matplotlib  
- **Data:** 10K+ dog images (Kaggle dataset)

---

## Project Structure

Dog-Vision/
│
├── static/
│ ├── uploaded/ # Uploaded images
│ └── model/ # Saved models (.keras/.h5)
│
├── templates/
│ └── index.html # Flask HTML template
│
├── app.py # Flask application
├── dog_breeds.json # List of dog breeds
├── README.md # This file
└── requirements.txt # Python dependencies


---

## Installation

1. **Clone the repo:**
```bash
git clone https://github.com/<your-username>/Dog-Vision.git
cd Dog-Vision
```

2. Create a virtual environment (recommended):
python -m venv venv
# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate

3. Running the Flask App
python app.py
Open your browser and go to: http://127.0.0.1:5000/


# Model Training

- **Dataset:** 10K+ dog images ([Kaggle Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification))  
- **Model:** MobileNetV2 feature extractor + Dense output layer (120 classes)  
- **Loss:** `categorical_crossentropy`  
- **Optimizer:** Adam  
- **Training:** 
  - Split into train/validation sets  
  - Early stopping and TensorBoard callbacks implemented  
- **Saved model formats:** `.keras`, `.h5`  

# Visualizing Predictions

- Visualize first 25 training images with labels  
- Plot predictions and top-10 probabilities for each image  
- **Color coding:**  
  - Green highlights correct prediction  
  - Red indicates mismatch

