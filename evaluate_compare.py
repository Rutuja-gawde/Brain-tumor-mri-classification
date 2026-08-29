"""
evaluate_compare.py

Loads the trained CNN, Random Forest, and SVM models, evaluates all three
on the held-out test set, and produces a side-by-side comparison: accuracy,
precision, recall, F1 (macro + weighted), confusion matrices, and a
bar chart comparing all three.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
)

from train_ml import extract_hog_features  # reuse the same HOG extractor

# ---------------- Config ----------------
PROCESSED_DIR = Path("processed")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
# -----------------------------------------


def evaluate_model(name, y_true, y_pred, classes):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    print(f"\n=== {name} ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro):    {recall:.4f}")
    print(f"F1 (macro):        {f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(f"{name} — Confusion Matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    fname = RESULTS_DIR / f"confusion_matrix_{name.lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150)
    plt.close()

    return {
        "model": name,
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(PROCESSED_DIR / "label_map.json") as f:
        label_map = json.load(f)
    classes = [label_map[str(i)] for i in range(len(label_map))]

    X_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")

    # ---- CNN ----
    print("Evaluating CNN...")
    cnn = tf.keras.models.load_model(MODELS_DIR / "cnn_model.h5")
    # NOTE: the model already has a Rescaling(1./255) layer built in, so
    # pass X_test in raw (0-255) — don't divide by 255 again here.
    cnn_probs = cnn.predict(X_test)
    cnn_preds = np.argmax(cnn_probs, axis=1)
    cnn_metrics = evaluate_model("CNN", y_test, cnn_preds, classes)

    # ---- Random Forest ----
    print("\nExtracting HOG features for test set...")
    X_test_hog = extract_hog_features(X_test)
    rf = joblib.load(MODELS_DIR / "rf_model.joblib")
    rf_preds = rf.predict(X_test_hog)
    rf_metrics = evaluate_model("Random Forest", y_test, rf_preds, classes)

    # ---- SVM ----
    # SVM was trained on scaled HOG features, so the same scaler must be
    # applied to the test features before prediction.
    print("\nEvaluating SVM...")
    scaler = joblib.load(MODELS_DIR / "hog_scaler.joblib")
    X_test_hog_scaled = scaler.transform(X_test_hog)
    svm = joblib.load(MODELS_DIR / "svm_model.joblib")
    svm_preds = svm.predict(X_test_hog_scaled)
    svm_metrics = evaluate_model("SVM", y_test, svm_preds, classes)

    # ---- Comparison table ----
    df = pd.DataFrame([cnn_metrics, rf_metrics, svm_metrics])
    df.to_csv(RESULTS_DIR / "comparison_report.csv", index=False)
    print("\n=== Comparison ===")
    print(df.to_string(index=False))

    # ---- Comparison bar chart ----
    metrics_to_plot = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    x = np.arange(len(metrics_to_plot))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, df.iloc[0][metrics_to_plot], width, label="CNN")
    ax.bar(x, df.iloc[1][metrics_to_plot], width, label="Random Forest")
    ax.bar(x + width, df.iloc[2][metrics_to_plot], width, label="SVM")
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1"])
    ax.set_ylim(0, 1)
    ax.set_title("CNN vs Random Forest vs SVM — Test Set Comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "comparison_report.png", dpi=150)

    print(f"\nSaved comparison table + chart + confusion matrices to '{RESULTS_DIR}/'")


if __name__ == "__main__":
    main()