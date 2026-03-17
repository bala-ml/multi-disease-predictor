import logging
from pathlib import Path

import pandas as pd
from joblib import load

from src.backend.config.settings import Settings
from src.utils.preprocess_util import replace_zeros_with_nan

# Load settings
settings = Settings()

DIABETES_MODEL_PATH = Path(settings.diabetes_model_path)
CARDIO_RISK_MODEL_PATH = Path(settings.cardio_risk_model_path)
LOG_PATH = Path(settings.log_path)

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH),
    ],
)


# Load models once
logging.info("Loading trained models...")
diabetes_model = load(DIABETES_MODEL_PATH)
cardio_risk_model = load(CARDIO_RISK_MODEL_PATH)
logging.info("Models loaded successfully.")


# common prediction function
def predict_disease(disease: str, input_data: dict):

    if disease == "diabetes":
        model = diabetes_model
    elif disease == "cardio_risk":
        model = cardio_risk_model
    else:
        raise ValueError("Invalid disease type. Use 'diabetes' or 'cardio_risk'")
    
    X_df = pd.DataFrame([input_data])

    prediction = int(model.predict(X_df)[0])

    # probability for positive class (class = 1)
    probability = float(model.predict_proba(X_df)[0][1])

    logging.info(
        f"[{disease}] prediction={prediction}, probability={probability}"
    )

    return {
        "disease": disease,
        "prediction": prediction,
        "probability": probability,
    }


# Example usage - comment out in production
# diabetes_input = {
#     "Pregnancies": 2,
#     "Glucose": 120,
#     "BloodPressure": 70,
#     "SkinThickness": 25,
#     "Insulin": 80,
#     "BMI": 28.5,
#     "DiabetesPedigreeFunction": 0.5,
#     "Age": 30
# }

# cardio_input = {
#     "age": 52,
#     "sex": 1,
#     "cp": 0,
#     "trestbps": 125,
#     "chol": 212,
#     "fbs": 0,
#     "restecg": 1,
#     "thalach": 168,
#     "exang": 0,
#     "oldpeak": 1.0,
#     "slope": 2,
#     "ca": 0,
#     "thal": 2
# }


# print(predict_disease("diabetes", diabetes_input))
# print(predict_disease("cardio_risk", cardio_input))