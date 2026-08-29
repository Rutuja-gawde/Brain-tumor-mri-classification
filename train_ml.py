"""
train_ml.py

Trains two classical ML classifiers on HOG (Histogram of Oriented Gradients)
features extracted from the MRI images: Random Forest and SVM. HOG is used
because raw pixels don't work well as input to classical ML models — HOG
captures edge/shape information in a much more compact, informative feature
vector.
"""

import time
from pathlib import Path

import numpy as np
import joblib
from skimage.feature import hog
from skimage.color import rgb2gray
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# ---------------- Config ----------------
PROCESSED_DIR = Path("processed")
MODELS_DIR = Path("models")
# -----------------------------------------


def extract_hog_features(X):
    """Converts a batch of RGB images into HOG feature vectors."""
    features = []
    for img in X:
        gray = rgb2gray(img)
        feat = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
        )
        features.append(feat)
    return np.array(features)


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")

    print("Extracting HOG features from training images (this can take a few minutes)...")
    t0 = time.time()
    X_train_hog = extract_hog_features(X_train)
    print(f"Done in {time.time() - t0:.1f}s. Feature vector length: {X_train_hog.shape[1]}")

    # cache features so re-runs (e.g. tuning) don't redo extraction
    np.save(PROCESSED_DIR / "X_train_hog.npy", X_train_hog)

    # ---------------- Random Forest ----------------
    print("\nTraining Random Forest (small grid search over key params)...")
    rf_param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [None, 30],
        "min_samples_split": [2, 5],
    }
    rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced")
    rf_search = GridSearchCV(rf, rf_param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=2)
    rf_search.fit(X_train_hog, y_train)

    print(f"\nBest RF params: {rf_search.best_params_}")
    print(f"Best RF CV accuracy: {rf_search.best_score_:.4f}")

    best_rf = rf_search.best_estimator_
    joblib.dump(best_rf, MODELS_DIR / "rf_model.joblib")
    print(f"Saved Random Forest model to {MODELS_DIR / 'rf_model.joblib'}")

    # ---------------- SVM ----------------
    # SVM is sensitive to feature scale (unlike Random Forest), so HOG
    # features are standardized first. The scaler is saved too since the
    # test set needs the same scaling applied before prediction.
    print("\nScaling features and training SVM (small grid search over key params)...")
    scaler = StandardScaler().fit(X_train_hog)
    X_train_scaled = scaler.transform(X_train_hog)
    joblib.dump(scaler, MODELS_DIR / "hog_scaler.joblib")

    svm_param_grid = {
        "C": [1, 10],
        "gamma": ["scale", "auto"],
        "kernel": ["rbf"],
    }
    # NOTE: we don't pass probability= at all — False is already SVC's
    # default, and sklearn fires its "probability is deprecated" warning
    # whenever this parameter is explicitly set (even to False), which
    # is misleading. Not passing it avoids the warning entirely while
    # keeping the exact same (fast) behavior.
    svm = SVC(class_weight="balanced", random_state=42)
    svm_search = GridSearchCV(svm, svm_param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=2)
    svm_search.fit(X_train_scaled, y_train)

    print(f"\nBest SVM params: {svm_search.best_params_}")
    print(f"Best SVM CV accuracy: {svm_search.best_score_:.4f}")

    best_svm = svm_search.best_estimator_
    joblib.dump(best_svm, MODELS_DIR / "svm_model.joblib")
    print(f"Saved SVM model to {MODELS_DIR / 'svm_model.joblib'}")


if __name__ == "__main__":
    main()