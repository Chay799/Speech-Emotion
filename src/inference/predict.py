import os
import numpy as np
from src.features.audio_loader import load_audio
from src.features.feature_extractor import extract_features
from src.models.sernet import build_sernet

FEATURE_SIZE = 680   # must match training
model = build_sernet(input_shape=(FEATURE_SIZE, 1), num_classes=3)
model.load_weights("model.weights.h5")

EMOTIONS = ["ANGRY 😠", "NEUTRAL 😐", "SAD 😢"]


def predict_emotion(audio_path):
    signal = load_audio(audio_path)
    features = extract_features(signal)

    features = np.expand_dims(features, axis=0)
    features = np.expand_dims(features, axis=2)

    preds = model.predict(features, verbose=0)[0]
    label = np.argmax(preds)

    return {
        "emotion": EMOTIONS[label],
        "confidence": float(preds[label]),
        "probabilities": {
            "angry": float(preds[0]),
            "neutral": float(preds[1]),
            "sad":   float(preds[2])
        }
    }


# ⭐ FIX for tests (old code expects tuple)
def predict_folder(folder_path):
    results = []

    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            path = os.path.join(folder_path, file)

            result = predict_emotion(path)

            results.append((file, result))

    return results