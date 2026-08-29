"""
app.py

Streamlit demo for the Brain Tumor MRI Classification project.
Upload an MRI scan and see live predictions from all three trained
models: CNN, Random Forest, and SVM.

Run with:
    streamlit run app.py

Expects these files to already exist (produced by the training scripts):
    models/cnn_model.h5
    models/rf_model.joblib
    models/svm_model.joblib
    models/hog_scaler.joblib
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import joblib
import tensorflow as tf
from skimage.feature import hog
from skimage.color import rgb2gray

# ---------------- Config ----------------
IMG_SIZE = 128
MODELS_DIR = Path("models")
# Class order matches the label indices used during training (see
# data_preprocessing.py) — hardcoded here rather than read from
# processed/label_map.json, since processed/ isn't pushed to deployment
# (it's just cached training arrays, regenerated locally).
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
# -----------------------------------------

st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="centered",
)


# ---------- Cached loaders (run once, reused across interactions) ----------
@st.cache_resource
def load_models():
    cnn = tf.keras.models.load_model(MODELS_DIR / "cnn_model.h5")
    rf = joblib.load(MODELS_DIR / "rf_model.joblib")
    svm = joblib.load(MODELS_DIR / "svm_model.joblib")
    scaler = joblib.load(MODELS_DIR / "hog_scaler.joblib")
    return cnn, rf, svm, scaler


def preprocess_image(pil_img: Image.Image):
    """Resize + convert to the same format used during training."""
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.uint8)  # raw 0-255, CNN rescales internally
    return arr


def extract_hog(img_arr: np.ndarray):
    gray = rgb2gray(img_arr)
    feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )
    return feat.reshape(1, -1)


# ---------------------------- UI ----------------------------
st.title("🧠 Brain Tumor MRI Classification")
st.caption(
    "Comparing CNN, Random Forest, and SVM on brain MRI scans — "
    "classifies into glioma, meningioma, pituitary tumor, or no tumor."
)

try:
    cnn_model, rf_model, svm_model, hog_scaler = load_models()
    classes = CLASSES
    models_ready = True
except Exception as e:
    models_ready = False
    st.error(
        "Couldn't load the trained models. Make sure you've run "
        "`data_preprocessing.py`, `train_cnn.py`, and `train_ml.py` first, "
        "and that this app is running from the project's root folder."
    )
    st.exception(e)

if models_ready:
    uploaded_file = st.file_uploader(
        "Upload a brain MRI scan (JPG or PNG)", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file)

        col1, col2 = st.columns([1, 1.4])
        with col1:
            st.image(pil_img, caption="Uploaded scan", use_container_width=True)

        img_arr = preprocess_image(pil_img)

        # ---- CNN prediction ----
        cnn_input = np.expand_dims(img_arr, axis=0)  # raw 0-255, model rescales internally
        cnn_probs = cnn_model.predict(cnn_input, verbose=0)[0]
        cnn_pred_idx = int(np.argmax(cnn_probs))

        # ---- Random Forest + SVM prediction (need HOG features) ----
        hog_feat = extract_hog(img_arr)
        rf_pred_idx = int(rf_model.predict(hog_feat)[0])
        rf_probs = rf_model.predict_proba(hog_feat)[0]

        hog_feat_scaled = hog_scaler.transform(hog_feat)
        svm_pred_idx = int(svm_model.predict(hog_feat_scaled)[0])
        svm_scores = svm_model.decision_function(hog_feat_scaled)[0]
        # SVM wasn't trained with probability calibration, so convert its
        # raw decision scores into a display-only pseudo-confidence via
        # softmax — this is NOT a calibrated probability, just for display.
        svm_softmax = np.exp(svm_scores - np.max(svm_scores))
        svm_softmax = svm_softmax / svm_softmax.sum()

        with col2:
            st.subheader("Predictions")
            st.metric("CNN", classes[cnn_pred_idx].capitalize(),
                       f"{cnn_probs[cnn_pred_idx]*100:.1f}% confidence")
            st.metric("Random Forest", classes[rf_pred_idx].capitalize(),
                       f"{rf_probs[rf_pred_idx]*100:.1f}% confidence")
            st.metric("SVM", classes[svm_pred_idx].capitalize(),
                       f"{svm_softmax[svm_pred_idx]*100:.1f}% confidence*")
            st.caption("*SVM confidence is an approximate score, not a calibrated probability.")

        st.divider()
        st.subheader("Full breakdown by class")

        breakdown_df = pd.DataFrame({
            "Class": [c.capitalize() for c in classes],
            "CNN": cnn_probs,
            "Random Forest": rf_probs,
            "SVM (approx.)": svm_softmax,
        }).set_index("Class")
        st.bar_chart(breakdown_df)

        if all(p == cnn_pred_idx for p in [rf_pred_idx, svm_pred_idx]):
            st.success(f"All three models agree: **{classes[cnn_pred_idx].capitalize()}**")
        else:
            st.warning("The models don't all agree — worth a closer look.")