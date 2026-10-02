"""Central paths and constants used across the project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DATA = ROOT / "data" / "raw" / "telco_churn.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

CHURN_MODEL_PATH = MODELS_DIR / "churn_model.joblib"
UPLIFT_MODEL_PATH = MODELS_DIR / "uplift_model.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "Churn"
ID_COL = "customerID"

NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "num_services",
    "avg_monthly_spend",
    "charge_increase",
]
CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "tenure_group",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Retention-offer economics (assumptions, documented in the README).
OFFER_DISCOUNT = 0.20       # 20% off the monthly bill...
OFFER_MONTHS = 3            # ...for 3 months
CLV_MONTHS = 12             # revenue horizon used to value a retained customer

for _d in (PROCESSED_DIR, MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
