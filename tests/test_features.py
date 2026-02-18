from src.data.dataset_loader import create_splits
from src.features.audio_loader import load_audio
from src.features.feature_extractor import extract_features

train_df, _ = create_splits()

sample = train_df.iloc[0]["path"]

audio = load_audio(sample)
features = extract_features(audio)

print("Feature vector shape:", features.shape)
