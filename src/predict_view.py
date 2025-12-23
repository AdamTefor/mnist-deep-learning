import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

def main():
    # 1) Charger le modèle
    model = keras.models.load_model("models/mnist_cnn.keras")

    # 2) Charger MNIST (test)
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()

    # 3) Choisir une image au hasard
    idx = np.random.randint(0, len(x_test))
    img = x_test[idx]
    true_label = y_test[idx]

    # 4) Prétraitement
    img_norm = img.astype("float32") / 255.0
    inp = np.expand_dims(img_norm, axis=(0, -1))

    # 5) Prédiction
    probs = model.predict(inp, verbose=0)[0]
    pred = int(np.argmax(probs))
    conf = float(np.max(probs))

    # 6) Affichage
    plt.imshow(img, cmap="gray")
    plt.title(f"Vrai: {true_label} | Prédit: {pred} | Confiance: {conf:.2f}")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
