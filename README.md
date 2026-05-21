# Offshore Well Anomaly Detector

Machine learning pipeline for anomaly detection in offshore oil well 
sensor data, built on the [Petrobras 3W Dataset](https://github.com/petrobras/3W).

## Overview

This repository contains the full source code for the diploma thesis:
**"Modelling and Prediction of Key Parameters of Oil and Gas 
Installations Using Machine Learning Based on Sensor Data"**  
*Timur Kasymbekov, AITU, 2026*

The pipeline implements:
- Group-aware train/test splitting (GroupShuffleSplit) to prevent well-level data leakage
- Sliding window feature extraction (60s window, 30s step, 24 features)
- Comparative study: Random Forest, XGBoost (Optuna-tuned), and LSTM baseline
- Three-tier recommendation system (NORMAL / WATCH / ANOMALY)
- SHAP TreeExplainer for per-prediction feature attribution
- Interactive Streamlit web application

## Results

| Model | ROC-AUC | FN | FP | F1-macro |
|-------|---------|----|----|----------|
| RF Baseline | 0.9616 | 11,649 | 6,182 | 0.8961 |
| RF Tuned ✓ | 0.9633 | **7,787** | 8,947 | 0.9027 |
| XGBoost Baseline | 0.9658 | 12,051 | 6,367 | 0.8926 |
| XGBoost Tuned | 0.9583 | 9,195 | 7,106 | 0.9051 |
| LSTM Baseline | 0.8795 | 11,702 | 14,397 | 0.85 |

RF Tuned selected as final model (lowest false negative count).

## Repository Structure
```
src/ # Shared modules (loader, preprocessor, features, config)
notebooks/        # Jupyter notebooks (run in order: 01 → 04)
outputs/          # Generated at runtime — see Google Drive link below
app.py            # Streamlit recommendation system
requirements.txt
```
## Quickstart

```bash
git clone https://github.com/Spikw285/DatasetViewer.git
cd DatasetViewer
pip install -r requirements.txt

# Download pre-computed artefacts from Google Drive (see Appendix A)
# Place in outputs/models/ and outputs/parquet_files/

streamlit run app.py
```

## Notebook Execution Order

| # | Notebook | Purpose |
|---|----------|---------|
| 01 | `01_eda.ipynb` | Exploratory data analysis |
| 02 | `02_features.ipynb` | Feature extraction, group split |
| 03_1 | `03_1_modeling_basic.ipynb` | RF and XGBoost baselines |
| 03_2 | `03_2_modeling_lstm.ipynb` | LSTM baseline (GPU required) |
| 03_3 | `03_3_modeling_experiments.ipynb` | Optuna tuning, SHAP analysis |
| 04 | `04_inference.ipynb` | Inference demo, threshold analysis |

## Data

Pre-computed feature matrices and trained models are available on 
[Google Drive](https://drive.google.com/drive/folders/10pCQGke0bFdZ4FZ5aXWz_jCgJFSfR5YK).  
Raw dataset: [Petrobras 3W GitHub](https://github.com/petrobras/3W)

## Environment

Python 3.12.13 · scikit-learn 1.6.1 · xgboost 3.2.0 · 
tensorflow 2.20.0 · shap 0.51.0 · optuna 4.8.0

## License

Source code: MIT  
Dataset: CC BY 4.0 (Petrobras)