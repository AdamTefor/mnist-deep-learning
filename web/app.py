import base64
import io
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, render_template, request, jsonify
from tensorflow import keras

app = Flask(__name__)

# Charger le modèle une seule fois au démarrage
model = keras.models.load_model("../models/best_mnist_cnn.keras")

def preprocess_pil(img: Image.Image) -> np.ndarray:
    """
    On transforme l'image canvas en format MNIST:
    - grayscale
    - resize 28x28
    - normalisation 0..1
    - shape (1, 28, 28, 1)
    """
    img = img.convert("L")  # gris
    img = ImageOps.fit(img, (28, 28), method=Image.Resampling.LANCZOS)

    arr = np.array(img).astype("float32")

    # Le canvas est souvent noir sur fond blanc -> inverser pour MNIST (blanc sur noir)
    if arr.mean() > 127:
        arr = 255.0 - arr

    arr = arr / 255.0
    arr = np.expand_dims(arr, axis=(0, -1))  # (1, 28, 28, 1)
    return arr

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "Image manquante"}), 400

    # data:image/png;base64,XXXX
    b64 = data["image"].split(",")[-1]
    image_bytes = base64.b64decode(b64)

    img = Image.open(io.BytesIO(image_bytes))
    x = preprocess_pil(img)

    probs = model.predict(x, verbose=0)[0]
    pred = int(np.argmax(probs))
    conf = float(np.max(probs))

    top3_idx = np.argsort(probs)[::-1][:3].tolist()
    top3 = [{"digit": int(i), "prob": float(probs[i])} for i in top3_idx]

    return jsonify({
        "prediction": pred,
        "confidence": conf,
        "top3": top3
    })

if __name__ == "__main__":
    app.run(debug=True)
