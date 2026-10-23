# server.py
# Serves the React frontend and a small prediction API.
# Run from the project folder:  python server.py     then open http://127.0.0.1:5000

import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
from tensorflow import keras
from class_names import class_names

here = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(here, "data")
model = keras.models.load_model(os.path.join(here, "models", "traffic_sign_mlp.keras"))

test_csv = os.path.join(data_folder, "Test.csv")
df_test = pd.read_csv(test_csv) if os.path.exists(test_csv) else None

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024   # 10 MB upload limit


def predict(img):
    # same steps as training: 3 channels, 32 x 32, pixels 0-1, batch of one
    img = img.convert("RGB").resize((32, 32))
    x = np.expand_dims(np.array(img, dtype="float32") / 255.0, axis=0)
    probs = model.predict(x, verbose=0)[0]
    top = np.argsort(probs)[::-1][:5]
    return {
        "class_id": int(top[0]),
        "sign": class_names[top[0]],
        "confidence": float(probs[top[0]]),
        "top5": [{"class_id": int(i), "name": class_names[i], "confidence": float(probs[i])} for i in top],
    }


@app.route("/")
def home():
    return send_from_directory("static", "index.html")


@app.route("/api/classes")
def classes():
    return jsonify(class_names)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    file = request.files.get("image")
    if file is None:
        return jsonify({"error": "No image received. Choose an image file."}), 400
    try:
        img = Image.open(file)
    except Exception:
        return jsonify({"error": "This file is not a readable image. Try a PNG or JPG."}), 400
    return jsonify(predict(img))


@app.route("/api/random")
def api_random():
    if df_test is None:
        return jsonify({"error": "data/Test.csv was not found, so random test images are unavailable."}), 404
    row = df_test.sample(1).iloc[0]
    img = Image.open(os.path.join(data_folder, row["Path"]))
    result = predict(img)
    true_id = int(row["ClassId"])
    result.update({
        "true_id": true_id,
        "true_name": class_names[true_id],
        "correct": result["class_id"] == true_id,
        "image_url": "/testimg/" + os.path.basename(row["Path"]),
    })
    return jsonify(result)


@app.route("/testimg/<name>")
def test_image(name):
    return send_from_directory(os.path.join(data_folder, "Test"), name)


@app.route("/meta/<int:class_id>")
def meta_image(class_id):
    return send_from_directory(os.path.join(data_folder, "Meta"), str(class_id) + ".png")


if __name__ == "__main__":
    app.run(port=5000, debug=False)
