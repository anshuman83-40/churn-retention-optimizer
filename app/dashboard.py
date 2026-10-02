"""Streamlit dashboard: score customers, plan a retention campaign, inspect the models."""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works without `pip install -e .`

from churn.config import FIGURES_DIR, RAW_DATA, REPORTS_DIR  # noqa: E402
from churn.predict import score  # noqa: E402

st.set_page_config(page_title="Churn & Retention Optimizer", page_icon="📉", layout="wide")

FRIENDLY = {
    "Contract": "Contract type", "tenure": "Months as customer", "tenure_group": "Tenure group",
    "InternetService": "Internet service", "PaymentMethod": "Payment method", "MonthlyCharges": "Monthly bill",
    "TotalCharges": "Total billed", "TechSupport": "Tech support", "OnlineSecurity": "Online security",
    "num_services": "Number of services", "avg_monthly_spend": "Avg monthly spend",
    "charge_increase": "Bill increase vs. avg", "PaperlessBilling": "Paperless billing",
}


@st.cache_data
def load_reports():
    churn = json.loads((REPORTS_DIR / "churn_metrics.json").read_text())
    uplift = json.loads((REPORTS_DIR / "uplift_metrics.json").read_text())
    return churn, uplift


@st.cache_data
def sample_customers(n=500):
    return pd.read_csv(RAW_DATA).drop(columns=["Churn"]).sample(n, random_state=7)


@st.cache_data(show_spinner="Scoring customers...")
def score_frame(df: pd.DataFrame) -> pd.DataFrame:
    res = pd.DataFrame(score(df.drop(columns=["customerID"], errors="ignore").to_dict("records")))
    out = pd.concat([df.reset_index(drop=True), res], axis=1)
    out["top_reason"] = res["top_risk_factors"].map(
        lambda r: FRIENDLY.get(r[0]["feature"], r[0]["feature"]) + f" = {r[0]['value']}" if r else "-")
    return out.drop(columns=["top_risk_factors"])


churn_rep, uplift_rep = load_reports()
st.title("📉 Churn & Retention Optimizer")
st.caption("Predicts who will churn, explains why, and decides **who is actually worth a retention offer** "
           "using causal uplift modelling, not just churn risk.")

tab1, tab2, tab3 = st.tabs(["🧍 Score a customer", "🎯 Campaign planner", "📊 Model insights"])

# ------------------------------------------------------------------ single customer
with tab1:
    with st.form("customer"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Account**")
            tenure = st.slider("Months as customer", 0, 72, 3)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            monthly = st.number_input("Monthly bill ($)", 18.0, 120.0, 85.0, step=1.0)
            payment = st.selectbox("Payment method", ["Electronic check", "Mailed check",
                                                      "Bank transfer (automatic)", "Credit card (automatic)"])
            paperless = st.radio("Paperless billing", ["Yes", "No"], horizontal=True)
        with c2:
            st.markdown("**Services**")
            internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
            no_net = internet == "No"
            opts = ["No internet service"] if no_net else ["No", "Yes"]
            security = st.selectbox("Online security", opts)
            backup = st.selectbox("Online backup", opts)
            protection = st.selectbox("Device protection", opts)
            support = st.selectbox("Tech support", opts)
            tv = st.selectbox("Streaming TV", opts)
            movies = st.selectbox("Streaming movies", opts)
        with c3:
            st.markdown("**Customer**")
            gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
            senior = st.radio("Senior citizen", ["No", "Yes"], horizontal=True)
            partner = st.radio("Partner", ["No", "Yes"], horizontal=True)
            dependents = st.radio("Dependents", ["No", "Yes"], horizontal=True)
            phone = st.radio("Phone service", ["Yes", "No"], horizontal=True)
            lines = st.selectbox("Multiple lines", ["No", "Yes"] if phone == "Yes" else ["No phone service"])
        submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

    if submitted:
        rec = dict(gender=gender, SeniorCitizen=senior, Partner=partner, Dependents=dependents, tenure=tenure,
                   PhoneService=phone, MultipleLines=lines, InternetService=internet, OnlineSecurity=security,
                   OnlineBackup=backup, DeviceProtection=protection, TechSupport=support, StreamingTV=tv,
                   StreamingMovies=movies, Contract=contract, PaperlessBilling=paperless, PaymentMethod=payment,
                   MonthlyCharges=monthly, TotalCharges=monthly * tenure)
        r = score([rec])[0]
        m1, m2, m3 = st.columns(3)
        m1.metric("Churn probability", f"{r['churn_probability']:.0%}", r["risk_level"].upper() + " risk",
                  delta_color="off")
        m2.metric("Offer reduces churn by", f"{r['offer_uplift'] * 100:+.1f} pp")
        m3.metric("Expected value of offer", f"${r['offer_expected_value_usd']:,.0f}")
        if r["recommendation"] == "send_offer":
            st.success(f"✅ **Send the retention offer.** {r['recommendation_reason']}")
        else:
            st.warning(f"🚫 **Don't send the offer.** {r['recommendation_reason']}")
        if r["top_risk_factors"]:
            st.markdown("**Why is this customer at risk?** (SHAP explanation)")
            for f in r["top_risk_factors"]:
                st.markdown(f"- {FRIENDLY.get(f['feature'], f['feature'])} = `{f['value']}` "
                            f"(+{f['impact']:.2f} log-odds)")

# ------------------------------------------------------------------ campaign
with tab2:
    st.markdown("Upload customers in the Telco CSV format, or use a sample of 500 real customers.")
    up = st.file_uploader("Customer CSV", type="csv")
    df = pd.read_csv(up).drop(columns=["Churn"], errors="ignore") if up else sample_customers()
    scored = score_frame(df)

    send = scored[scored["recommendation"] == "send_offer"]
    everyone = scored["offer_expected_value_usd"].sum()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers", f"{len(scored):,}")
    k2.metric("High risk", f"{(scored['risk_level'] == 'high').sum():,}")
    k3.metric("Recommended offers", f"{len(send):,}", f"{len(send) / len(scored):.0%} of base", delta_color="off")
    k4.metric("Expected campaign value", f"${send['offer_expected_value_usd'].sum():,.0f}",
              f"${send['offer_expected_value_usd'].sum() - everyone:+,.0f} vs. offer to all")
    st.caption(f"Sending the offer to all {len(scored):,} customers would be worth ${everyone:,.0f}: discounts get "
               f"wasted on customers who would stay anyway or who can't be persuaded.")

    cols = ["customerID", "churn_probability", "risk_level", "top_reason", "offer_uplift",
            "offer_expected_value_usd", "recommendation", "Contract", "tenure", "MonthlyCharges"]
    cols = [c for c in cols if c in scored]
    st.dataframe(scored[cols].sort_values("offer_expected_value_usd", ascending=False),
                 use_container_width=True, hide_index=True,
                 column_config={"churn_probability": st.column_config.ProgressColumn(
                     "Churn prob.", min_value=0, max_value=1, format="%.2f")})
    st.download_button("Download target list (CSV)", send[cols].to_csv(index=False), "retention_targets.csv")

# ------------------------------------------------------------------ insights
with tab3:
    xgb = churn_rep["metrics"]["xgboost"]
    lr = churn_rep["metrics"]["logistic_regression"]
    pv = uplift_rep["scenarios"][uplift_rep["serving_scenario"]]["policy_value"]
    risk20, uplift20 = pv["Top 20% by churn risk (traditional)"], pv["Top 20% by predicted uplift"]
    a, b, c, d = st.columns(4)
    a.metric("ROC-AUC (test)", f"{xgb['roc_auc']:.3f}", f"{xgb['roc_auc'] - lr['roc_auc']:+.3f} vs LogReg")
    b.metric("PR-AUC (test)", f"{xgb['pr_auc']:.3f}")
    c.metric("Top-decile lift", f"{churn_rep['top_decile_lift']:.1f}x")
    d.metric("Uplift vs risk targeting (same budget)", f"{uplift20 / risk20:.1f}x value",
             f"${uplift20:,.0f} vs ${risk20:,.0f}", delta_color="off")

    st.subheader("Churn model")
    g1, g2 = st.columns(2)
    g1.image(str(FIGURES_DIR / "model_roc_pr.png"))
    g2.image(str(FIGURES_DIR / "shap_importance.png"))

    st.subheader("Uplift model (who to target)")
    g3, g4 = st.columns(2)
    g3.image(str(FIGURES_DIR / "uplift_policy_value.png"))
    g4.image(str(FIGURES_DIR / "uplift_qini.png"))
    st.dataframe(pd.DataFrame(uplift_rep["policies"]), use_container_width=True, hide_index=True)
    st.markdown("**When is uplift modelling worth it?** If the riskiest customers are also the most persuadable "
                "(*risk-aligned*), plain risk targeting is already near-optimal. If many risky customers are lost "
                "causes or sleeping dogs (*realistic*), risk targeting wastes the budget and uplift wins.")
    s1, _ = st.columns([3, 2])
    s1.image(str(FIGURES_DIR / "uplift_scenarios.png"))
    st.info("The uplift experiment is semi-synthetic: real customers and churn baselines, with a simulated "
            "randomized offer whose true effect is known. This lets us measure targeting quality exactly. "
            "See the README for details.")

    st.subheader("EDA")
    e1, e2, e3 = st.columns(3)
    e1.image(str(FIGURES_DIR / "eda_contract.png"))
    e2.image(str(FIGURES_DIR / "eda_tenure_group.png"))
    e3.image(str(FIGURES_DIR / "eda_paymentmethod.png"))
