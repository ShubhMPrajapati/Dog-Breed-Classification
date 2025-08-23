from flask import Flask, request, render_template
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.saving import register_keras_serializable

# -------------------------------
# Model & Feature Extractor Setup
# -------------------------------

# Load pre-trained MobileNetV2 as feature extractor
feature_extractor_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

@register_keras_serializable()
def apply_feature_extractor(x):
    x = preprocess_input(x)
    return feature_extractor_model(x)

# Model file paths
MODEL_KERAS = "static/model/dog_model.keras"
MODEL_H5 = "static/model/dog_model.h5"
MODEL_FOLDER = "static/model"
##
print()
# Load class labels
with open("static/model/dog_breeds.json", "r") as f:
    class_labels = json.load(f)

# Load the model
model = None
if os.path.isfile(MODEL_KERAS):
    print("Loading model from Keras (.keras)...")
    model = tf.keras.models.load_model(MODEL_KERAS)
elif os.path.isfile(MODEL_H5):
    print("Loading model from HDF5 (.h5)...")
    model = tf.keras.models.load_model(MODEL_H5, compile=False)
elif os.path.isdir(MODEL_FOLDER) and os.path.isfile(os.path.join(MODEL_FOLDER, "saved_model.pb")):
    print("Loading model from TensorFlow SavedModel format...")
    model = tf.keras.models.load_model(MODEL_FOLDER)
else:
    raise FileNotFoundError(
        "No valid model found! Please save as .keras, .h5, or SavedModel format."
    )

# -------------------------------
# Flask App Setup
# -------------------------------

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if file:
            # Save uploaded file
            filepath = os.path.join("static", file.filename)
            file.save(filepath)

            # Preprocess image
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            # Predict breed
            preds = model.predict(img_array)
            class_idx = np.argmax(preds, axis=1)[0]
            breed_name = class_labels[class_idx]

            return render_template(
                "index.html",
                prediction=f"Predicted Breed: {breed_name}",
                img_path=file.filename
            )

    return render_template("index.html")

# -------------------------------
# Run Flask App
# -------------------------------

if __name__ == "__main__":
    app.run(debug=False)
