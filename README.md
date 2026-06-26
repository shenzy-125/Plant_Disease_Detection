# Plant_Disease_Detection
# 🌿 Plant Disease Detection using Transfer Learning

Classifies plant leaf diseases from images using **EfficientNetB0** (pre-trained on ImageNet), fine-tuned on the PlantVillage dataset across **15 disease classes** covering Tomato, Potato, and Pepper plants.

---

## Dataset

[PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) — images organized into folders by disease class. 15 classes total across 3 plant types.

---

## Model

Transfer Learning with **EfficientNetB0**, trained in two phases:

- **Phase 1 (20 epochs)** — Base frozen, only classification head trained
- **Phase 2 (10 epochs)** — Last 20 layers unfrozen, fine-tuned at lr=1e-5

Best weights saved via `ModelCheckpoint` on `val_accuracy`.

---

## Evaluation

- Classification Report (Precision, Recall, F1 Score)
- Confusion Matrix
- Accuracy & Loss curves (both phases combined)

---

## Setup

```bash
pip install tensorflow opencv-python scikit-learn matplotlib seaborn pandas numpy
```

Place the PlantVillage dataset in a folder named `PlantVillage/` in the root, then run `Plant_Disease_Detection.ipynb`.

To skip training, load the saved model directly:
```python
from tensorflow.keras.models import load_model
model = load_model('best_model.keras')
```

---

## Tech Stack
Python · TensorFlow/Keras · EfficientNetB0 · OpenCV · Scikit-Learn · Matplotlib

---

**Author:** Your Name · B.Tech CSE · [GitHub](https://github.com/your-username) · [LinkedIn](https://linkedin.com/in/your-username)
