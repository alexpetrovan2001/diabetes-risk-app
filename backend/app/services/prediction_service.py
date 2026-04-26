from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[3]
MODEL_PATH = BASE_DIR / "models" / "diabetes_model.joblib"
FEATURES_PATH = BASE_DIR / "models" / "feature_columns.joblib"

model = None
feature_columns = None


def load_artifacts():
    global model, feature_columns

    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)

    if feature_columns is None:
        if not FEATURES_PATH.exists():
            raise FileNotFoundError(
                f"Feature columns file not found at: {FEATURES_PATH}"
            )
        feature_columns = joblib.load(FEATURES_PATH)


def predict_diabetes_risk(payload):
    load_artifacts()

    input_data = pd.DataFrame(
        [
            {
                "Pregnancies": payload.pregnancies,
                "Glucose": payload.glucose,
                "BloodPressure": payload.blood_pressure,
                "SkinThickness": payload.skin_thickness,
                "Insulin": payload.insulin,
                "BMI": payload.bmi,
                "DiabetesPedigreeFunction": payload.diabetes_pedigree_function,
                "Age": payload.age,
            }
        ]
    )

    input_data = input_data[feature_columns]

    prediction = int(model.predict(input_data)[0])

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(input_data)[0][1])
    else:
        probability = float(prediction)

    risk_label = "high" if prediction == 1 else "low"
    message = (
        "Predicted elevated diabetes risk."
        if prediction == 1
        else "Predicted lower diabetes risk."
    )

    return {
        "prediction": prediction,
        "risk_label": risk_label,
        "probability": round(probability, 4),
        "message": message,
        "model_version": "logistic-regression-v1",
    }