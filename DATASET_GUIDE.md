# Dataset setup

This project does **not** include medical images. Use only a public dataset whose terms allow your intended educational use, and read its license and privacy terms first.

## Expected folder layout

Put images into class-named folders like this:

```text
data/
└── images/
    ├── NORMAL/
    │   ├── normal_001.jpeg
    │   └── normal_002.jpeg
    └── PNEUMONIA/
        ├── pneumonia_001.jpeg
        └── pneumonia_002.jpeg
```

The class names are examples. Use labels that match the dataset's own documentation. Keep the dataset out of GitHub unless its license explicitly permits redistribution.

## Public dataset starting point

Search for the **Chest X-Ray Images (Pneumonia)** dataset on Kaggle and review the dataset page, source, terms, and known limitations before downloading. Dataset labels can be noisy and the dataset may contain duplicates or biases. Do not treat model outputs as diagnoses.

After placing images into the structure above:

```powershell
python train_model.py
```

The script creates `models/image_classifier.joblib` and `reports/evaluation.json`.

## Important evaluation caveats

- This is a simple HOG + Linear SVM baseline, not a modern clinical imaging model.
- The random image split may leak information if related images/patients appear in both splits. Prefer patient-level splits when metadata supports them.
- Accuracy alone is insufficient; inspect class-wise precision/recall, data balance, external validation, and dataset bias.
- Never use this student project for screening, treatment decisions, or patient care.
