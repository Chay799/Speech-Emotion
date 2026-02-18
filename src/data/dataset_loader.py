import os
import pandas as pd
from sklearn.model_selection import train_test_split

EMOTION_MAP = {
    "angry": 0,
    "neutral": 1,
    "sad": 2
}

LANGUAGES = ["telugu", "kannada"]


def create_splits(test_size=0.2):
    rows = []

    # 🔒 SAFE DATASET SCAN (no junk files)
    for language in LANGUAGES:
        lang_path = os.path.join("data/raw", language)
        if not os.path.exists(lang_path):
            continue

        for emotion in EMOTION_MAP.keys():
            emotion_path = os.path.join(lang_path, emotion)
            if not os.path.exists(emotion_path):
                continue

            for file in os.listdir(emotion_path):
                if file.endswith(".wav"):
                    rows.append({
                        "path": os.path.join(emotion_path, file),
                        "language": language,
                        "emotion": emotion,
                        "label": EMOTION_MAP[emotion]
                    })

    df = pd.DataFrame(rows)

    print("\n🎵 TOTAL DATASET:", len(df))

    # 🔥 STRATIFIED SPLIT (language + emotion)
    df["stratify_col"] = df["language"] + "_" + df["emotion"]

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        stratify=df["stratify_col"]
    )

    train_df = train_df.drop(columns=["stratify_col"])
    test_df  = test_df.drop(columns=["stratify_col"])

    # 📊 Show distribution
    print("\n📊 TRAIN DISTRIBUTION")
    print(train_df.groupby(["language","emotion"]).size())

    print("\n📊 TEST DISTRIBUTION")
    print(test_df.groupby(["language","emotion"]).size())

    print("\nTrain samples:", len(train_df))
    print("Test samples:", len(test_df))

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
