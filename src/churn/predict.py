"""Inference: churn risk + per-customer reasons + retention-offer recommendation."""
from functools import lru_cache

import joblib
import numpy as np
import shap

from .config import CHURN_MODEL_PATH, MIN_OFFER_VALUE, UPLIFT_MODEL_PATH
from .data import prepare_input
from .features import aggregate_by_feature, original_feature_groups
from .uplift import offer_economics


@lru_cache(maxsize=1)
def load_models():
    churn = joblib.load(CHURN_MODEL_PATH)
    uplift = joblib.load(UPLIFT_MODEL_PATH)
    pipe = churn["pipeline"]
    explainer = shap.TreeExplainer(pipe.named_steps["clf"])
    groups = original_feature_groups(pipe.named_steps["pre"])
    return churn, uplift, explainer, groups


def risk_level(p: float) -> str:
    return "high" if p >= 0.6 else "medium" if p >= 0.3 else "low"


def recommend(uplift: float, value: float, p0: float) -> tuple[str, str]:
    if value > MIN_OFFER_VALUE:
        return "send_offer", "Offer is expected to keep this customer and pay for itself."
    if uplift < -0.01:
        return "do_not_send", "Offer may backfire: likely to prompt this customer to reconsider."
    if p0 < 0.15:
        return "do_not_send", "Customer is likely to stay anyway; discount would be wasted."
    return "do_not_send", "At-risk, but a discount won't change their decision. Try a service fix instead."


def score(records: list[dict], top_k: int = 3) -> list[dict]:
    churn, uplift_art, explainer, groups = load_models()
    X = prepare_input(records)
    pipe = churn["pipeline"]
    proba = pipe.predict_proba(X)[:, 1]

    sv = explainer.shap_values(pipe.named_steps["pre"].transform(X))
    learner = uplift_art["learner"]
    p0_hat, p1_hat = learner.predict_outcomes(X)
    u_hat = learner.predict_uplift(X)
    ev = offer_economics(X, p1_hat, u_hat)

    results = []
    for i in range(len(X)):
        contrib = aggregate_by_feature(sv[i], groups)
        drivers = sorted(contrib.items(), key=lambda kv: -kv[1])
        reasons = [
            {"feature": f, "value": _fmt(X.iloc[i][f]), "impact": round(v, 3)}
            for f, v in drivers[:top_k] if v > 0
        ]
        action, why = recommend(float(u_hat[i]), float(ev[i]), float(p0_hat[i]))
        results.append({
            "churn_probability": round(float(proba[i]), 4),
            "risk_level": risk_level(float(proba[i])),
            "predicted_churn": bool(proba[i] >= churn["threshold"]),
            "top_risk_factors": reasons,
            "offer_uplift": round(float(u_hat[i]), 4),
            "offer_expected_value_usd": round(float(ev[i]), 2),
            "recommendation": action,
            "recommendation_reason": why,
        })
    return results


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return round(float(v), 2)
    if isinstance(v, np.integer):
        return int(v)
    return v
