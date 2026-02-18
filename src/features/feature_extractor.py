import numpy as np
import librosa

SR = 16000
DURATION = 4   # seconds


# =====================================================
# Helper to extract statistics from time features
# =====================================================
def stats(feature):
    mean = np.mean(feature, axis=1)
    std  = np.std(feature, axis=1)
    return np.hstack([mean, std])


# =====================================================
# MFCC  (speech shape)
# =====================================================
def extract_mfcc(signal, sr=SR, n_mfcc=40):
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)

    # dynamics (VERY IMPORTANT for neutral)
    delta1 = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    return np.hstack([
        stats(mfcc),
        stats(delta1),
        stats(delta2)
    ])


# =====================================================
# MEL SPECTROGRAM (energy + loudness)
# =====================================================
def extract_mel(signal, sr=SR, n_mels=64):
    mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel)

    delta1 = librosa.feature.delta(mel_db)
    delta2 = librosa.feature.delta(mel_db, order=2)

    return np.hstack([
        stats(mel_db),
        stats(delta1),
        stats(delta2)
    ])


# =====================================================
# PROSODIC FEATURES (pitch + energy)
# =====================================================
def extract_prosodic(signal, sr=SR):

    # Pitch
    pitch, _ = librosa.piptrack(y=signal, sr=sr)
    pitch = pitch[pitch > 0]
    pitch_mean = np.mean(pitch) if len(pitch) > 0 else 0
    pitch_std  = np.std(pitch) if len(pitch) > 0 else 0

    # Energy
    energy = librosa.feature.rms(y=signal)[0]
    energy_mean = np.mean(energy)
    energy_std  = np.std(energy)

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(signal)[0]
    zcr_mean = np.mean(zcr)
    zcr_std  = np.std(zcr)

    return np.array([
        pitch_mean, pitch_std,
        energy_mean, energy_std,
        zcr_mean, zcr_std
    ])


# =====================================================
# CHROMA (voice tone / mood)
# =====================================================
def extract_chroma(signal, sr=SR):
    stft = np.abs(librosa.stft(signal))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    return stats(chroma)


# =====================================================
# SPECTRAL CONTRAST (tension / harshness)
# =====================================================
def extract_spectral_contrast(signal, sr=SR):
    contrast = librosa.feature.spectral_contrast(y=signal, sr=sr)
    return stats(contrast)


# =====================================================
# TONNETZ (harmonic emotion)
# =====================================================
def extract_tonnetz(signal, sr=SR):
    harmonic = librosa.effects.harmonic(signal)
    tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
    return stats(tonnetz)


# =====================================================
# MAIN FUNCTION (pipeline compatible)
# =====================================================
def extract_features(signal):
    """
    Final feature vector for Angry / Neutral / Sad
    """

    # Fix length to 4 sec
    max_len = SR * DURATION
    if len(signal) < max_len:
        signal = np.pad(signal, (0, max_len - len(signal)))
    else:
        signal = signal[:max_len]

    mfcc      = extract_mfcc(signal)
    mel       = extract_mel(signal)
    prosodic  = extract_prosodic(signal)
    chroma    = extract_chroma(signal)
    contrast  = extract_spectral_contrast(signal)
    tonnetz   = extract_tonnetz(signal)

    features = np.hstack([
        mfcc,
        mel,
        prosodic,
        chroma,
        contrast,
        tonnetz
    ])

    return features
