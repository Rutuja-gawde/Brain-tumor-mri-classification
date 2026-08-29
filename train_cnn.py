"""
train_cnn.py

Trains a CNN on the preprocessed brain tumor MRI dataset (from
data_preprocessing.py) and saves the trained model + training history.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ---------------- Config ----------------
PROCESSED_DIR = Path("processed")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
IMG_SIZE = 128
NUM_CLASSES = 4
BATCH_SIZE = 32
EPOCHS = 25
# -----------------------------------------


def build_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Rescaling(1. / 255),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    X_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")

    # carve out a validation split from training data
    val_split = int(0.85 * len(X_train))
    idx = np.random.RandomState(42).permutation(len(X_train))
    X_train, X_val = X_train[idx[:val_split]], X_train[idx[val_split:]]
    y_train, y_val = y_train[idx[:val_split]], y_train[idx[val_split:]]

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    model = build_cnn()
    model.summary()

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stop, reduce_lr],
    )

    model.save(MODELS_DIR / "cnn_model.h5")
    with open(RESULTS_DIR / "cnn_history.json", "w") as f:
        json.dump(history.history, f)

    # quick test-set accuracy for a sanity check (full comparison happens
    # in evaluate_compare.py)
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nCNN test accuracy: {test_acc:.4f}")

    # plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("CNN Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("CNN Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "cnn_training_curves.png", dpi=150)
    print(f"Saved model to {MODELS_DIR / 'cnn_model.h5'}")
    print(f"Saved training curves to {RESULTS_DIR / 'cnn_training_curves.png'}")


if __name__ == "__main__":
    main()