import numpy as np
from tensorflow import keras

def main():
    # 1) Charger le modèle sauvegardé
    model = keras.models.load_model("models/mnist_cnn.keras")

    # 2) Charger le dataset MNIST (partie test)
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()

    # 3) Choisir une image au hasard
    index = np.random.randint(0, len(x_test))
    image = x_test[index].astype("float32") / 255.0

    # 4) Adapter la forme pour le CNN : (28,28) → (1,28,28,1)
    image = np.expand_dims(image, axis=(0, -1))

    # 5) Faire la prédiction
    probabilities = model.predict(image, verbose=0)[0]
    prediction = np.argmax(probabilities)
    confidence = np.max(probabilities)

    # 6) Afficher les résultats
    print("Index image :", index)
    print("Vrai chiffre :", y_test[index])
    print("Chiffre prédit :", prediction)
    print("Confiance :", round(confidence, 4))

if __name__ == "__main__":
    main()
