import pandas as pd

from churn.config import FEATURES, TARGET
from churn.data import add_features, clean, load_dataset, prepare_input


def test_dataset_shape_and_target():
    df = load_dataset()
    assert len(df) == 7043
    assert set(df[TARGET].unique()) == {0, 1}
    assert df[FEATURES].isna().sum().sum() == 0


def test_blank_total_charges_become_zero():
    raw = pd.DataFrame([{"TotalCharges": " ", "SeniorCitizen": 0, "Churn": "No"}])
    out = clean(raw)
    assert out["TotalCharges"].iloc[0] == 0.0
    assert out["SeniorCitizen"].iloc[0] == "No"


def test_engineered_features(customer):
    df = add_features(clean(pd.DataFrame([customer])))
    row = df.iloc[0]
    assert row["tenure_group"] == "0-6m"
    assert row["num_services"] == 4  # phone + fiber + streaming TV + tech support
    assert abs(row["charge_increase"]) < 1e-9  # TotalCharges = tenure x monthly


def test_prepare_input_fills_total_charges(customer):
    customer = {k: v for k, v in customer.items() if k != "TotalCharges"}
    X = prepare_input([customer])
    assert list(X.columns) == FEATURES
    assert X["TotalCharges"].iloc[0] == customer["tenure"] * customer["MonthlyCharges"]
