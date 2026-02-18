import librosa
import numpy as np

TARGET_SR = 22050
DURATION = 4  # seconds
SAMPLES = TARGET_SR * DURATION


def load_audio(file_path):
    """
    Load audio and ensure fixed length (3 sec)
    """
    signal, sr = librosa.load(file_path, sr=TARGET_SR)

    if len(signal) > SAMPLES:
        signal = signal[:SAMPLES]
    else:
        padding = SAMPLES - len(signal)
        signal = np.pad(signal, (0, padding))

    return signal
