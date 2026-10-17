from pathlib import Path
import json
import joblib
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.image_utils import image_to_features

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "images"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
ALLOWED = {".png", ".jpg", ".jpeg", ".bmp"}

def load_dataset():
    if not DATA_DIR.exists():
        raise SystemExit(
            f"Dataset folder not found: {DATA_DIR}\n"
            "Read DATASET_GUIDE.md and create data/images/<class_name>/ folders."
        )
    X, y = [], []
    for class_dir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
        for path in class_dir.rglob("*"):
            if path.suffix.lower() not in ALLOWED:
                continue
            try:
                with Image.open(path) as image:
                    X.append(image_to_features(image)[0])
                    y.append(class_dir.name)
            except Exception as exc:
                print(f"Skipping unreadable image {path}: {exc}")
    if len(X) < 10:
        raise SystemExit(f"Found only {len(X)} readable images. Add more labelled images before training.")
    labels, counts = np.unique(y, return_counts=True)
    if len(labels) < 2:
        raise SystemExit("At least two class folders with images are required.")
    if min(counts) < 2:
        raise SystemExit("Each class needs at least two images for a train/test split.")
    return np.asarray(X), np.asarray(y)

def main():
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = make_pipeline(StandardScaler(), LinearSVC(class_weight="balanced", random_state=42))
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "classes": sorted(set(y))}, MODEL_DIR / "image_classifier.joblib")
    report = {
        "warning": "Educational dataset evaluation only; not clinical validation.",
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=sorted(set(y))).tolist(),
        "labels_order": sorted(set(y)),
        "train_images": int(len(y_train)),
        "test_images": int(len(y_test)),
    }
    (REPORT_DIR / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Model saved to: {MODEL_DIR / 'image_classifier.joblib'}")
    print(f"Test accuracy: {report['test_accuracy']:.3f}")
    print("Evaluation report saved to reports/evaluation.json")
    print("Reminder: this is a student baseline, not a clinical diagnostic tool.")

if __name__ == "__main__":
    main()
