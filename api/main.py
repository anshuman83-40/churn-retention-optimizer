"""FastAPI service: POST customer data, get churn risk + retention recommendation."""
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from churn.predict import load_models, score

YesNo = Literal["Yes", "No"]
YesNoNoInternet = Literal["Yes", "No", "No internet service"]

@asynccontextmanager
async def lifespan(_app):
    load_models()  # warm the models once at startup
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Churn & Retention Optimizer",
    description="Predicts customer churn, explains why, and recommends whether a retention offer is worth it.",
    version="1.0.0",
)


class Customer(BaseModel):
    gender: Literal["Male", "Female"] = "Female"
    SeniorCitizen: YesNo = "No"
    Partner: YesNo = "No"
    Dependents: YesNo = "No"
    tenure: int = Field(3, ge=0, le=100, description="Months with the company")
    PhoneService: YesNo = "Yes"
    MultipleLines: Literal["Yes", "No", "No phone service"] = "No"
    InternetService: Literal["DSL", "Fiber optic", "No"] = "Fiber optic"
    OnlineSecurity: YesNoNoInternet = "No"
    OnlineBackup: YesNoNoInternet = "No"
    DeviceProtection: YesNoNoInternet = "No"
    TechSupport: YesNoNoInternet = "Yes"
    StreamingTV: YesNoNoInternet = "Yes"
    StreamingMovies: YesNoNoInternet = "No"
    Contract: Literal["Month-to-month", "One year", "Two year"] = "Month-to-month"
    PaperlessBilling: YesNo = "Yes"
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ] = "Electronic check"
    MonthlyCharges: float = Field(85.0, ge=0)
    TotalCharges: float | None = Field(None, ge=0, description="Defaults to tenure x MonthlyCharges")


class RiskFactor(BaseModel):
    feature: str
    value: str | float | int
    impact: float


class Prediction(BaseModel):
    churn_probability: float
    risk_level: Literal["low", "medium", "high"]
    predicted_churn: bool
    top_risk_factors: list[RiskFactor]
    offer_uplift: float
    offer_expected_value_usd: float
    recommendation: Literal["send_offer", "do_not_send"]
    recommendation_reason: str


@app.get("/health")
def health():
    churn, uplift, _, _ = load_models()
    return {"status": "ok", "churn_model": churn["model_name"], "uplift_model": uplift["name"]}


def _records(customers: list[Customer]) -> list[dict]:
    recs = []
    for c in customers:
        d = c.model_dump()
        if d["TotalCharges"] is None:
            d["TotalCharges"] = d["tenure"] * d["MonthlyCharges"]
        recs.append(d)
    return recs


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    return score(_records([customer]))[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(customers: list[Customer]):
    return score(_records(customers)) if customers else []
