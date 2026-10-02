"""Loading, cleaning and feature engineering for the Telco churn dataset."""
import numpy as np
import pandas as pd

from .config import FEATURES, ID_COL, RAW_DATA, TARGET

SERVICE_COLS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def load_raw(path=RAW_DATA) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # TotalCharges is blank for brand-new customers (tenure == 0) -> they have paid nothing yet.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"}).fillna(df["SeniorCitizen"])
    if TARGET in df.columns:
        df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineered features. Works on a single-row frame too (used by the API)."""
    df = df.copy()
    services = df[SERVICE_COLS].eq("Yes").sum(axis=1)
    internet = df["InternetService"].ne("No").astype(int)
    df["num_services"] = services + internet
    tenure = df["tenure"].clip(lower=1)
    df["avg_monthly_spend"] = np.where(df["tenure"] > 0, df["TotalCharges"] / tenure, df["MonthlyCharges"])
    # >0 means the customer now pays more than their historical average (price-increase signal).
    df["charge_increase"] = df["MonthlyCharges"] - df["avg_monthly_spend"]
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 6, 12, 24, 48, np.inf],
        labels=["0-6m", "7-12m", "13-24m", "25-48m", "49m+"],
    ).astype(str)
    return df


def load_dataset() -> pd.DataFrame:
    """Raw CSV -> cleaned, feature-engineered frame with id, features and target."""
    df = add_features(clean(load_raw()))
    return df[[ID_COL] + FEATURES + [TARGET]]


def prepare_input(records: list[dict]) -> pd.DataFrame:
    """Turn raw customer dicts (same schema as the CSV) into model-ready features."""
    df = pd.DataFrame(records)
    if "TotalCharges" not in df or df["TotalCharges"].isna().all():
        df["TotalCharges"] = df["MonthlyCharges"] * df["tenure"]
    df = add_features(clean(df))
    return df[FEATURES]
