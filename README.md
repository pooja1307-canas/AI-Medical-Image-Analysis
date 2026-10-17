# AI-Powered Medical Image Analysis System

A beginner-friendly, local student portfolio project demonstrating a medical-image classification workflow with Streamlit, image feature extraction, scikit-learn, and evaluation reports.

> **Safety notice:** This is an educational prototype, not a medical device and not suitable for clinical diagnosis or patient-care decisions.

## Features

- Local Streamlit image upload interface
- Image dimensions and preview
- Baseline HOG feature extraction + Linear SVM classifier
- Dataset-driven training (no fake prediction when a model is missing)
- Evaluation JSON report with accuracy, class-wise metrics, and confusion matrix
- Public-dataset setup guide and tests

## Requirements

- Windows 10/11
- Python 3.12 recommended
- PowerShell

## Setup (Windows PowerShell)

Open PowerShell in this folder and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks environment activation, run this in the same terminal and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run the dashboard

```powershell
streamlit run app.py
```

The dashboard opens locally in your browser. Before training, it will tell you that no model is available.

## Train a model

1. Read [`DATASET_GUIDE.md`](DATASET_GUIDE.md).
2. Add permitted public dataset images under `data/images/<label>/`.
3. Train:

```powershell
python train_model.py
```

4. Start or refresh the dashboard:

```powershell
streamlit run app.py
```

Training creates `models/image_classifier.joblib` and `reports/evaluation.json`.

## Run tests

```powershell
python -m pytest -q
```

## Suggested GitHub proof

- Add screenshots of the running dashboard and the evaluation report.
- Add a short explanation of dataset source, license, limitations, and evaluation.
- Do not upload dataset images unless redistribution is explicitly permitted.
- Do not commit `.venv/`, model binaries, private data, or large datasets.

## Limitations

This is a simple educational baseline, not a validated clinical system. It has no patient-level validation, external validation, DICOM support, or clinical workflow integration. Scores from the model are not calibrated clinical probabilities.
