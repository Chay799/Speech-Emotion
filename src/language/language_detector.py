import torch
import librosa
from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor

MODEL_NAME = "facebook/mms-lid-126"

print("Loading language detection model...")

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME)

LANG_MAP = {
    "tel": "Telugu",
    "kan": "Kannada"
}


def detect_language(audio_path):
    # load audio
    speech, sr = librosa.load(audio_path, sr=16000)

    inputs = feature_extractor(
        speech,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_id = torch.argmax(logits, dim=-1).item()
    lang_code = model.config.id2label[predicted_id]

    return LANG_MAP.get(lang_code, lang_code)
