# MNIST Deep Learning — Handwritten Digit Recognition (CNN)

Projet de Deep Learning qui reconnaît les chiffres manuscrits (0–9) à partir d’images, entraîné sur le dataset MNIST avec un réseau CNN (TensorFlow/Keras).  
Le projet inclut :
- Entraînement + évaluation (accuracy, matrice de confusion, classification report)
- Scripts de prédiction
- Visualisation d’une prédiction
- Mini application web (HTML/JavaScript + Flask) permettant de dessiner un chiffre sur un canvas et obtenir une prédiction

## Résultats
- Accuracy test : ~0.99 (variable selon l’entraînement)
- Meilleur modèle sauvegardé via ModelCheckpoint : `models/best_mnist_cnn.keras`

## Technologies
- Python 3.10
- TensorFlow / Keras
- NumPy, scikit-learn, matplotlib
- Flask (backend)
- HTML + JavaScript (frontend canvas)

## Structure du projet
mnist-deep-learning/
├─ src/
│ ├─ train.py
│ ├─ predict.py
│ ├─ predict_view.py
├─ models/
│ ├─ best_mnist_cnn.keras
│ ├─ mnist_cnn_final.keras
├─ web/
│ ├─ app.py
│ ├─ templates/
│ │ └─ index.html
│ └─ static/
│ └─ app.js
├─ requirements.txt
└─ README.md


## Installation (Windows)
1) Cloner le projet
```bash
git clone <URL_DU_REPO>
cd mnist-deep-learning


Créer et activer l’environnement virtuel

python -m venv .venv
.\.venv\Scripts\Activate.ps1


Installer les dépendances

pip install -r requirements.txt

Entraîner le modèle
python .\src\train.py


Sorties attendues :

Accuracy sur test affichée dans le terminal

Sauvegarde des modèles dans models/

Prédire sur une image test (console)
python .\src\predict.py

Visualiser une prédiction (image + résultat)
python .\src\predict_view.py

Lancer l’application Web (HTML/JS + Flask)

Aller dans le dossier web

cd web
python app.py


Ouvrir le navigateur :

http://127.0.0.1:5000

Fonctionnement :

Dessiner un chiffre sur le canvas

Cliquer sur "Prédire"

Le backend Flask envoie l’image au modèle TensorFlow et renvoie la prédiction + confiance

Notes

Le fichier .keras est un fichier binaire (modèle sauvegardé), il ne doit pas être ouvert comme un fichier texte.

Pour un usage production, utiliser un serveur WSGI (gunicorn/waitress) au lieu du serveur Flask de dev.

Auteur: Adam Tefor