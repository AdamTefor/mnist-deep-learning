import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report

def main():
    # 1) Charger MNIST
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # 2) Prétraitement : 0..255 -> 0..1
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0

    # 3) Ajouter le canal (28,28) -> (28,28,1) pour le CNN
    x_train = np.expand_dims(x_train, -1)
    x_test  = np.expand_dims(x_test, -1)

    # 4) Construire un CNN un peu plus "robuste"
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),  # évite l'overfitting (apprentissage "par cœur")

        layers.Dense(10, activation="softmax")
    ])

    # 5) Compiler
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 6) Callbacks (pro)
    os.makedirs("models", exist_ok=True)

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath="models/best_mnist_cnn.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=2,               # si ça n'améliore pas pendant 2 epochs => stop
        restore_best_weights=True,
        verbose=1
    )

    # 7) Entraîner (on met 20, mais EarlyStopping arrête avant si inutile)
    history = model.fit(
        x_train, y_train,
        epochs=20,
        batch_size=128,
        validation_split=0.1,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )

    # 8) Évaluer sur le vrai test
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nAccuracy sur test: {test_acc:.4f}")

    # 9) Rapport (utile pour apprendre)
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    print("\nMatrice de confusion:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred))

    # 10) Sauvegarder aussi une version "finale"
    model.save("models/mnist_cnn_final.keras")
    print("\nModèle final sauvegardé : models/mnist_cnn_final.keras")
    print("Meilleur modèle sauvegardé : models/best_mnist_cnn.keras")

if __name__ == "__main__":
    main()
