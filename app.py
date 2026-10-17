
from pathlib import Path

import streamlit as st
import numpy as np
from PIL import Image
import joblib

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "image_classifier.joblib"
REPORT_PATH = ROOT / "reports" / "evaluation.json"

st.set_page_config(
    page_title="AI Medical Image Analysis",
    page_icon="🩻",
    layout="wide",
)

st.title("🩻 AI-Powered Medical Image Analysis System")
st.caption(
    "Educational research prototype | Local demo | Not for clinical diagnosis"
)

with st.sidebar:
    st.header("📋 Project Information")
    st.write("**Task:** Chest X-ray image classification")
    st.write("**Features:** HOG image features")
    st.write("**Classifier:** Scikit-learn model")
    st.write("**Dataset:** Public chest X-ray images")

    st.warning(
        "This student prototype is not a medical device. "
        "Do not use its output for patient care."
    )

    st.divider()
    st.subheader("System Status")

    if MODEL_PATH.exists():
        st.success("Trained model available")
    else:
        st.error("Trained model not found")

# Dashboard metrics
st.subheader("📊 Model Overview")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Model Status",
    "Ready" if MODEL_PATH.exists() else "Not trained",
)

col2.metric(
    "Input Format",
    "PNG / JPG / JPEG",
)

col3.metric(
    "Processing",
    "Local",
)

st.divider()

st.subheader("🔍 Upload Chest X-ray Image")
st.write(
    "Upload an image to see the model's predicted dataset label. "
    "The prediction is experimental and is not a medical diagnosis."
)

uploaded = st.file_uploader(
    "Choose a chest X-ray image",
    type=["png", "jpg", "jpeg"],
)

if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")

        left, right = st.columns(2)

        with left:
            st.subheader("Uploaded Image")
            st.image(image, use_container_width=True)
            st.caption(
                f"Dimensions: {image.width} × {image.height} pixels"
            )

        with right:
            st.subheader("Prediction Result")

            if MODEL_PATH.exists():
                bundle = joblib.load(MODEL_PATH)
                from src.image_utils import image_to_features

                features = image_to_features(image)
                model = bundle["model"]
                prediction = model.predict(features)[0]

                st.success(f"Predicted dataset label: {prediction}")

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(features)[0]
                    classes = list(model.classes_)

                    st.write("Model scores")
                    st.bar_chart(
                        {
                            str(label): float(score)
                            for label, score in zip(
                                classes, probabilities
                            )
                        }
                    )

                    st.caption(
                        "Scores are model outputs, not calibrated "
                        "clinical probabilities."
                    )

                st.info(
                    "This prediction is for educational purposes only. "
                    "It cannot confirm or rule out pneumonia. "
                    "Consult a qualified healthcare professional."
                )

            else:
                st.error("No trained model found.")
                st.code("python train_model.py")

    except Exception as exc:
        st.error(f"Could not process the image: {exc}")

st.divider()

# Evaluation report
st.subheader("📈 Model Evaluation")

if REPORT_PATH.exists():
    try:
        import json

        with open(REPORT_PATH, "r", encoding="utf-8") as file:
            report = json.load(file)

        st.write("Evaluation information saved by the training script:")
        st.json(report)

    except (OSError, ValueError) as exc:
        st.warning(f"Could not read evaluation report: {exc}")
else:
    st.info(
        "Evaluation report not found. Train the model to generate it."
    )

st.divider()

st.subheader("🔄 Project Workflow")

st.markdown(
    """
    1. **Dataset preparation:** Use labelled public images.
    2. **Model training:** Train the baseline classifier.
    3. **Evaluation:** Review the saved evaluation report.
    4. **Image testing:** Try sample images cautiously.
    5. **Responsible use:** Never treat predictions as a diagnosis.
    """
)

st.caption(
    "Privacy: this app processes uploaded images locally. "
    "Avoid identifiable patient data. This prototype is not validated "
    "for clinical use."
)
