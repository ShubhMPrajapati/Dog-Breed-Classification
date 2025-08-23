from flask import Flask, request, render_template
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2
from keras.saving import register_keras_serializable

# -------------------------------
# Upload folder
# -------------------------------
UPLOAD_FOLDER = "static/uploaded"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------------
# Keras serializable (if needed)
# -------------------------------
@register_keras_serializable()
def apply_feature_extractor(x):
    x = preprocess_input(x)
    return x

# -------------------------------
# Model & labels paths
# -------------------------------
MODEL_KERAS = "static/model/dog_model.keras"
MODEL_H5 = "static/model/dog_model.h5"
MODEL_FOLDER = "static/model"

with open("static/model/dog_breeds.json", "r") as f:
    class_labels = json.load(f)

# -------------------------------
# Load trained classifier model
# -------------------------------
model = None
if os.path.isfile(MODEL_KERAS):
    print("Loading model from .keras format...")
    model = tf.keras.models.load_model(MODEL_KERAS, compile=False)
elif os.path.isfile(MODEL_H5):
    print("Loading model from .h5 format...")
    model = tf.keras.models.load_model(MODEL_H5, compile=False)
elif os.path.isdir(MODEL_FOLDER) and os.path.isfile(os.path.join(MODEL_FOLDER, "saved_model.pb")):
    print("Loading model from SavedModel format...")
    model = tf.keras.models.load_model(MODEL_FOLDER, compile=False)
else:
    raise FileNotFoundError(
        "No valid model found! Please save as .keras, .h5, or SavedModel format."
    )

# -------------------------------
# Load feature extractor (MobileNetV2)
# -------------------------------
feature_extractor = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224, 224, 3)
)

# -------------------------------
# Flask app setup
# -------------------------------
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    img_path = None

    if request.method == "POST":
        file = request.files.get("file")
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Preprocess image
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # Extract features using MobileNetV2
            features = feature_extractor.predict(img_array)

            # Predict using your classifier
            preds = model.predict(features)
            class_idx = np.argmax(preds, axis=1)[0]
            breed_name = class_labels[class_idx]

            prediction = f"Predicted Breed: {breed_name}"
            img_path = f"uploaded/{file.filename}"

    return render_template("index.html", prediction=prediction, img_path=img_path)

# -------------------------------
# Run Flask app on Render
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
