from flask import Flask, request, render_template
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.saving import register_keras_serializable
from io import BytesIO
from PIL import Image
import base64

# -------------------------------
# Register custom function
# -------------------------------
feature_extractor_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224, 224, 3)
)

@register_keras_serializable()
def apply_feature_extractor(x):
    x = preprocess_input(x)
    return feature_extractor_model(x)

# -------------------------------
# Model & labels paths
# -------------------------------
MODEL_PATH = "static/model/dog_model.keras"

with open("static/model/dog_breeds.json", "r") as f:
    class_labels = json.load(f)

# -------------------------------
# Load trained classifier model
# -------------------------------
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# -------------------------------
# Flask app
# -------------------------------
app = Flask(__name__, static_folder="static")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    img_data = None

    if request.method == "POST":
        file = request.files.get("file")
        if file:
            # Read and preprocess image from memory
            img_bytes = file.read()
            img = Image.open(BytesIO(img_bytes)).resize((224, 224))
            img_array = np.expand_dims(image.img_to_array(img), axis=0)

            # Predict
            preds = model.predict(img_array)
            class_idx = np.argmax(preds, axis=1)[0]
            breed_name = class_labels[class_idx]

            prediction = f"Predicted Breed: {breed_name}"

            # Convert image to base64 for inline display (ensure JPEG format)
            buffered = BytesIO()
            img.convert("RGB").save(buffered, format="JPEG")
            img_data = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return render_template("index.html", prediction=prediction, img_data=img_data)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(port=port, debug=False)
