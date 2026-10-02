"""Scores the whole customer base once and serves dashboard data (KPIs, segments, tables)."""
import json
from functools import lru_cache

import numpy as np
import pandas as pd

from .config import CLV_MONTHS, RAW_DATA, REPORTS_DIR
from .predict import score

SEGMENTS = {
    "Persuadable": "Will likely stay if they get the offer: target them.",
    "Lost Cause": "High risk, but a discount won't change their mind. Fix the service issue instead.",
    "Sleeping Dog": "Loyal today; an offer may prompt them to shop around. Do not contact.",
    "Loyal": "Low churn risk. No retention spend needed.",
    "Monitor": "Moderate risk, offer not worth it yet. Watch for changes.",
}

SERVICE_LABELS = {
    "OnlineSecurity": "online security", "OnlineBackup": "online backup",
    "DeviceProtection": "device protection", "TechSupport": "tech support",
}


def describe_factor(feature: str, value) -> str:
    """Human-readable label for a SHAP risk factor."""
    if feature == "Contract":
        return f"{value} contract"
    if feature in ("tenure", "tenure_group"):
        return f"New customer ({value}{' mo' if feature == 'tenure' else ''})"
    if feature == "InternetService":
        return f"{value} internet"
    if feature == "PaymentMethod":
        return f"Pays by {str(value).lower()}"
    if feature == "MonthlyCharges":
        return f"High bill (${value:.0f}/mo)" if isinstance(value, (int, float)) else "High bill"
    if feature in ("TotalCharges", "avg_monthly_spend"):
        return "Low lifetime spend"
    if feature == "charge_increase":
        return "Recent price increase"
    if feature in SERVICE_LABELS:
        return f"No {SERVICE_LABELS[feature]}" if value == "No" else SERVICE_LABELS[feature].capitalize()
    if feature == "PaperlessBilling":
        return "Paperless billing"
    if feature == "num_services":
        return f"Only {value} services" if isinstance(value, (int, float)) and value <= 2 else f"{value} services"
    if feature == "SeniorCitizen":
        return "Senior citizen"
    return f"{feature}: {value}"


def segment_of(r: dict) -> str:
    if r["recommendation"] == "send_offer":
        return "Persuadable"
    if r["offer_uplift"] < -0.01:
        return "Sleeping Dog"
    if r["churn_probability"] >= 0.5:
        return "Lost Cause"
    if r["churn_probability"] < 0.15:
        return "Loyal"
    return "Monitor"


def actions_for(row: dict, seg: str) -> list[str]:
    acts = []
    if seg == "Persuadable":
        acts.append("Send 20% discount offer (3 months)")
    if seg == "Sleeping Dog":
        return ["Do not contact: offer may backfire", "Keep service quality high"]
    if row["Contract"] == "Month-to-month" and row["churn_probability"] >= 0.3:
        acts.append("Offer 1-year contract upgrade")
    if row["TechSupport"] == "No" and row["InternetService"] != "No" and row["churn_probability"] >= 0.3:
        acts.append("Proactive tech-support call")
    if row["PaymentMethod"] == "Electronic check" and row["churn_probability"] >= 0.3:
        acts.append("Nudge to switch to auto-pay")
    if seg == "Loyal":
        acts.append("No retention spend: consider cross-sell")
    return acts or ["Monitor monthly"]


@lru_cache(maxsize=1)
def customer_base() -> pd.DataFrame:
    raw = pd.read_csv(RAW_DATA)
    actual = (raw.pop("Churn") == "Yes").astype(int)
    raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="coerce").fillna(0.0)
    raw["SeniorCitizen"] = raw["SeniorCitizen"].map({0: "No", 1: "Yes"})
    res = score(raw.drop(columns=["customerID"]).to_dict("records"), top_k=4)
    df = pd.concat([raw, pd.DataFrame(res)], axis=1)
    df["actual_churn"] = actual
    df["segment"] = [segment_of(r) for r in res]
    df["revenue_at_risk"] = (df["churn_probability"] * df["MonthlyCharges"] * CLV_MONTHS).round(2)
    df["factors"] = [
        [{"label": describe_factor(f["feature"], f["value"]), "feature": f["feature"], "impact": f["impact"]}
         for f in r["top_risk_factors"]]
        for r in res
    ]
    df["main_factors"] = df["factors"].map(lambda fs: ", ".join(f["label"] for f in fs[:2]) or "None significant")
    df["actions"] = [actions_for(row, seg) for row, seg in zip(df.to_dict("records"), df["segment"])]
    df["primary_action"] = df["actions"].map(lambda a: a[0])
    return df


def _reports():
    churn = json.loads((REPORTS_DIR / "churn_metrics.json").read_text())
    uplift = json.loads((REPORTS_DIR / "uplift_metrics.json").read_text())
    return churn, uplift


def summary() -> dict:
    df = customer_base()
    churn_rep, uplift_rep = _reports()
    n = len(df)
    pers = df[df["segment"] == "Persuadable"]

    seg_counts = df["segment"].value_counts()
    segments = [
        {"name": s, "count": int(seg_counts.get(s, 0)), "share": round(float(seg_counts.get(s, 0)) / n, 4),
         "description": d}
        for s, d in SEGMENTS.items()
    ]

    imp = churn_rep["feature_importance"]
    merged: dict[str, float] = {}
    labels = {"Contract": "Contract type", "tenure": "Tenure", "tenure_group": "Tenure",
              "InternetService": "Internet service", "OnlineSecurity": "Online security",
              "TechSupport": "Tech support", "PaymentMethod": "Payment method",
              "MonthlyCharges": "Monthly bill", "PaperlessBilling": "Paperless billing",
              "TotalCharges": "Lifetime spend", "avg_monthly_spend": "Lifetime spend"}
    for f, v in imp.items():
        key = labels.get(f, f)
        merged[key] = merged.get(key, 0) + v
    top = sorted(merged.items(), key=lambda kv: -kv[1])[:5]
    total_imp = sum(merged.values())
    factors = [{"name": k, "share": round(v / total_imp, 4)} for k, v in top]

    bins = [-1, 6, 12, 18, 24, 36, 48, 60, 72]
    names = ["0-6", "7-12", "13-18", "19-24", "25-36", "37-48", "49-60", "61-72"]
    g = df.groupby(pd.cut(df["tenure"], bins=bins, labels=names), observed=True)
    by_tenure = {
        "labels": names,
        "actual": [round(float(v), 4) for v in g["actual_churn"].mean()],
        "predicted": [round(float(v), 4) for v in g["churn_probability"].mean()],
    }

    pv = uplift_rep["scenarios"][uplift_rep["serving_scenario"]]["policy_value"]
    xgb = churn_rep["metrics"]["xgboost"]
    return {
        "kpis": {
            "total_customers": n,
            "predicted_churn_rate": round(float(df["churn_probability"].mean()), 4),
            "actual_churn_rate": round(float(df["actual_churn"].mean()), 4),
            "high_risk": int((df["risk_level"] == "high").sum()),
            "revenue_at_risk": round(float(df["revenue_at_risk"].sum()), 0),
            "monthly_revenue": round(float(df["MonthlyCharges"].sum()), 0),
            "customers_to_save": int(len(pers)),
            "campaign_value": round(float(pers["offer_expected_value_usd"].sum()), 0),
        },
        "segments": segments,
        "top_factors": factors,
        "churn_by_tenure": by_tenure,
        "model": {
            "roc_auc": xgb["roc_auc"], "pr_auc": xgb["pr_auc"], "f1": xgb["f1"], "recall": xgb["recall"],
            "precision": xgb["precision"], "brier": xgb["brier"], "lift": churn_rep["top_decile_lift"],
            "baseline_roc_auc": churn_rep["metrics"]["logistic_regression"]["roc_auc"],
            "uplift_learner": uplift_rep["selected_learner"],
            "uplift_vs_risk": {"uplift": pv["Top 20% by predicted uplift"],
                               "risk": pv["Top 20% by churn risk (traditional)"],
                               "everyone": pv["Offer to everyone"],
                               "oracle": pv["Oracle (true effect, upper bound)"]},
            "policies": uplift_rep["policies"],
            "scenarios": {k: v["headline"] for k, v in uplift_rep["scenarios"].items()},
        },
    }


TABLE_COLS = ["customerID", "Contract", "tenure", "MonthlyCharges", "churn_probability", "risk_level", "segment",
              "main_factors", "primary_action", "offer_expected_value_usd", "revenue_at_risk"]
FILTERS = {"high": lambda d: d["risk_level"] == "high", **{s.lower().replace(" ", "_"): (lambda s: lambda d: d["segment"] == s)(s) for s in SEGMENTS}}


def query(segment: str = "all", q: str = "", sort: str = "churn_probability", desc: bool = True) -> pd.DataFrame:
    df = customer_base()
    if segment in FILTERS:
        df = df[FILTERS[segment](df)]
    if q:
        ql = q.lower()
        mask = (df["customerID"].str.lower().str.contains(ql, regex=False)
                | df["segment"].str.lower().str.contains(ql, regex=False)
                | df["main_factors"].str.lower().str.contains(ql, regex=False)
                | df["Contract"].str.lower().str.contains(ql, regex=False))
        df = df[mask]
    if sort in df.columns:
        df = df.sort_values(sort, ascending=not desc)
    return df


def page(segment="all", q="", sort="churn_probability", desc=True, page_no=1, size=10) -> dict:
    df = query(segment, q, sort, desc)
    total = len(df)
    start = (max(page_no, 1) - 1) * size
    rows = df.iloc[start:start + size][TABLE_COLS]
    return {"total": total, "page": page_no, "size": size,
            "rows": json.loads(rows.to_json(orient="records"))}


def customer(customer_id: str) -> dict | None:
    df = customer_base()
    hit = df[df["customerID"] == customer_id]
    if hit.empty:
        return None
    r = hit.iloc[0]
    profile_cols = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "Contract", "PaymentMethod",
                    "PaperlessBilling", "MonthlyCharges", "TotalCharges", "PhoneService", "MultipleLines",
                    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                    "StreamingTV", "StreamingMovies"]
    out = {c: (r[c].item() if isinstance(r[c], np.generic) else r[c]) for c in profile_cols}
    out.update({
        "customerID": r["customerID"],
        "churn_probability": float(r["churn_probability"]),
        "risk_level": r["risk_level"],
        "segment": r["segment"],
        "segment_description": SEGMENTS[r["segment"]],
        "factors": r["factors"],
        "offer_uplift": float(r["offer_uplift"]),
        "offer_expected_value_usd": float(r["offer_expected_value_usd"]),
        "revenue_at_risk": float(r["revenue_at_risk"]),
        "recommendation": r["recommendation"],
        "recommendation_reason": r["recommendation_reason"],
        "actions": list(r["actions"]),
    })
    return out
