import pytest


@pytest.fixture
def customer():
    return {
        "gender": "Female", "SeniorCitizen": "No", "Partner": "No", "Dependents": "No", "tenure": 3,
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "Yes",
        "StreamingTV": "Yes", "StreamingMovies": "No", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 85.0,
        "TotalCharges": 255.0,
    }


@pytest.fixture
def loyal_customer(customer):
    return {**customer, "tenure": 60, "Contract": "Two year", "InternetService": "DSL",
            "PaymentMethod": "Credit card (automatic)", "MonthlyCharges": 55.0, "TotalCharges": 3300.0}
