import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from tensorflow import keras

st.set_page_config(page_title="MNIST Digit Recognition", layout="centered")

@st.cache_resource
def load_model():
    return keras.models.load_model("models/mnist_cnn.keras")

def preprocess_image(img: Image.Image) -> np.ndarray:
    """
    Convertit une image uploadée en format compatible MNIST:
    - grayscale
    - inversion si fond blanc
    - resize 28x28
    - normalisation 0..1
    - shape (1, 28, 28, 1)
    """
    # Convertir en niveaux de gris
    img = img.convert("L")

    # Redimensionner en 28x28
    img = ImageOps.fit(img, (28, 28), method=Image.Resampling.LANCZOS)

    # Convertir en array
    arr = np.array(img).astype("float32")

    # Heuristique: si l'image est "claire" (fond blanc), on inverse (MNIST: chiffre blanc sur fond noir)
    if arr.mean() > 127:
        arr = 255.0 - arr

    # Normaliser
    arr = arr / 255.0

    # Ajouter dimensions
    arr = np.expand_dims(arr, axis=(0, -1))  # (1, 28, 28, 1)
    return arr

def main():
    st.title("Reconnaissance de chiffres (MNIST) — Demo")
    st.write("Upload une image d’un chiffre (0-9) et le modèle va prédire le chiffre.")

    model = load_model()

    uploaded = st.file_uploader("Choisir une image (png/jpg/jpeg)", type=["png", "jpg", "jpeg"])

    if uploaded is None:
        st.info("Astuce: utilise une image simple (chiffre bien centré).")
        return

    img = Image.open(uploaded)
    st.subheader("Image originale")
    st.image(img, use_container_width=True)

    x = preprocess_image(img)

    probs = model.predict(x, verbose=0)[0]
    pred = int(np.argmax(probs))
    conf = float(np.max(probs))

    st.subheader("Résultat")
    st.metric("Chiffre prédit", pred)
    st.write(f"Confiance: **{conf:.4f}**")

    # Afficher top 3 probabilités
    top3 = np.argsort(probs)[::-1][:3]
    st.subheader("Top 3 probabilités")
    for k in top3:
        st.write(f"{k} : {probs[k]:.4f}")

if __name__ == "__main__":
    main()
