# Brain Tumor MRI Classification — CNN vs Random Forest vs SVM

Classifies brain MRI scans into 4 classes: **glioma, meningioma, no tumor, pituitary**,
comparing three approaches:

1. **CNN** (deep learning) — 93.1% test accuracy
2. **SVM** (classical ML on scaled HOG features) — 90.8% test accuracy
3. **Random Forest** (classical ML on HOG features) — 83.8% test accuracy

## Dataset

[Brain Tumor MRI Dataset — Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

## Project Structure

- `data_preprocessing.py` — loads images, resizes, splits train/test
- `train_cnn.py` — trains the CNN
- `train_ml.py` — trains Random Forest + SVM on HOG features
- `evaluate_compare.py` — evaluates all three models, generates comparison
- `results_comparison.ipynb` — notebook version, loads trained models, shows charts
- `app.py` —interactive web app for live predictions
- `requirements.txt`
- `models/` — trained models (tracked via Git LFS, large files)

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Download the dataset from the Kaggle link above and place it as `data/Training/`
and `data/Testing/`, each with subfolders: `glioma/`, `meningioma/`, `notumor/`,
`pituitary/`.

## Run the pipeline

```bash
python data_preprocessing.py
python train_cnn.py
python train_ml.py
python evaluate_compare.py
```

## Run the app

```bash
streamlit run app.py
```

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| CNN | 93.1% | 93.6% | 93.1% | 92.9% |
| SVM | 90.8% | 91.5% | 90.8% | 90.4% |
| Random Forest | 83.8% | 85.0% | 83.8% | 83.1% |

"No tumor" was classified perfectly by all three models; glioma was the most
frequently confused class across all three.
