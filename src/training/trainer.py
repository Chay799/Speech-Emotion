import numpy as np
from tqdm import tqdm

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.data.dataset_loader import create_splits
from src.features.audio_loader import load_audio
from src.features.feature_extractor import extract_features
from src.models.sernet import build_sernet
from src.models.losses import get_loss, get_optimizer


# =========================================================
# Convert dataframe → feature tensors
# =========================================================
def build_feature_dataset(df):
    X, y = [], []

    print("Extracting features from audio files...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        audio = load_audio(row["path"])
        features = extract_features(audio)

        X.append(features)
        y.append(row["label"])

    X = np.array(X)
    y = np.array(y)

    # CNN expects (samples, timesteps, channels)
    X = np.expand_dims(X, axis=2)

    return X, y


# =========================================================
# MAIN TRAINING FUNCTION
# =========================================================
def train_model():

    # 1️⃣ Split dataset (balanced stratified split)
    train_df, test_df = create_splits()

    # 2️⃣ Extract features
    X_train, y_train = build_feature_dataset(train_df)
    X_test, y_test = build_feature_dataset(test_df)

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    # 3️⃣ Build SER-Net model
    input_shape = X_train.shape[1:]
    model = build_sernet(input_shape=input_shape, num_classes=3)

    model.compile(
        optimizer=get_optimizer(),
        loss=get_loss(),
        metrics=["accuracy"]
    )

    # =====================================================
    # 🔥 VERY IMPORTANT: FORCE MODEL TO LEARN ANGRY
    # =====================================================
    class_weights = {
        0: 4.0,   # ANGRY (strong boost)
        1: 3.0,   # NEUTRAL
        2: 2.5    # SAD
    }

    # =====================================================
    # 🔥 SMART TRAINING (AUTO STOP + LR SCHEDULER)
    # =====================================================
    callbacks = [
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=6,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # =====================================================
    # 4️⃣ Train model
    # =====================================================
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,        # will stop automatically early
        batch_size=8,      # better gradient stability
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # =====================================================
    # 5️⃣ Save trained weights
    # =====================================================
    model.save_weights("model.weights.h5")
    print("\n✅ Training finished. Weights saved as model.weights.h5")
    