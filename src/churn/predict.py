"""Inference: churn risk + per-customer reasons + retention-offer recommendation.

Two interchangeable model sets:
  "own"      ChurnUpliftNet, the from-scratch NumPy network (risk, uplift and explanations from one model)
  "library"  XGBoost churn model + SHAP + the best library uplift learner
"""
from functools import lru_cache

import joblib
import numpy as np
import shap

from .config import CHURN_MODEL_PATH, MIN_OFFER_VALUE, OWN_MODEL_PATH, UPLIFT_MODEL_PATH
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


MODELS = ("own", "library")
DEFAULT_MODEL = "own"


@lru_cache(maxsize=1)
def load_own():
    return joblib.load(OWN_MODEL_PATH)


def model_names() -> dict:
    churn, uplift, _, _ = load_models()
    return {"own": load_own()["name"], "library": f"{'XGBoost' if churn['model_name'] == 'xgboost' else 'Logistic Regression'} + {uplift['name']}"}


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


def score(records: list[dict], top_k: int = 3, model: str = DEFAULT_MODEL) -> list[dict]:
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; choose one of {MODELS}")
    X = prepare_input(records)
    if model == "own":
        art = load_own()
        net, threshold = art["model"], art["threshold"]
        p0_hat, p1_hat = net.predict_outcomes(X)
        proba, u_hat = p0_hat, p0_hat - p1_hat
        contribs = net.explain(X)  # occlusion, implemented in ownmodel.py
    else:
        churn, uplift_art, explainer, groups = load_models()
        pipe, threshold = churn["pipeline"], churn["threshold"]
        proba = pipe.predict_proba(X)[:, 1]
        sv = explainer.shap_values(pipe.named_steps["pre"].transform(X))
        contribs = [aggregate_by_feature(row, groups) for row in sv]
        learner = uplift_art["learner"]
        p0_hat, p1_hat = learner.predict_outcomes(X)
        u_hat = learner.predict_uplift(X)
    ev = offer_economics(X, p1_hat, u_hat)

    results = []
    for i in range(len(X)):
        contrib = contribs[i]
        drivers = sorted(contrib.items(), key=lambda kv: -kv[1])
        reasons = [
            {"feature": f, "value": _fmt(X.iloc[i][f]), "impact": round(v, 3)}
            for f, v in drivers[:top_k] if v > 0
        ]
        action, why = recommend(float(u_hat[i]), float(ev[i]), float(p0_hat[i]))
        results.append({
            "churn_probability": round(float(proba[i]), 4),
            "risk_level": risk_level(float(proba[i])),
            "predicted_churn": bool(proba[i] >= threshold),
            "top_risk_factors": reasons,
            "offer_uplift": round(float(u_hat[i]), 4),
            "offer_expected_value_usd": round(float(ev[i]), 2),
            "recommendation": action,
            "recommendation_reason": why,
            "contributions": {f: round(v, 4) for f, v in contrib.items()},
        })
    return results


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return round(float(v), 2)
    if isinstance(v, np.integer):
        return int(v)
    return v
