"""Uplift modelling: who should receive a retention offer?

The Telco dataset has no record of past offers, so we run a *semi-synthetic* randomized
experiment on top of the real customers:

  1. p0(x)  = leakage-free (out-of-fold) churn probability from the churn model  -> realistic baselines
  2. tau(x) = heterogeneous offer effect defined by plausible business rules      -> known ground truth
  3. T ~ Bernoulli(0.5), churn ~ Bernoulli(p0 if T=0 else p1)                      -> observed experiment

Uplift learners only see (X, T, Y) like in a real A/B test; the known tau lets us measure
exactly how good their targeting is. Uplift here = P(churn | no offer) - P(churn | offer).
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from .config import (
    CHURN_MODEL_PATH,
    CLV_MONTHS,
    FEATURES,
    FIGURES_DIR,
    OFFER_DISCOUNT,
    OFFER_MONTHS,
    PROCESSED_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET,
    UPLIFT_MODEL_PATH,
)
from .data import load_dataset
from .learners import SLearner, TLearner, XLearner

# --------------------------------------------------------------------------- simulation
SCENARIOS = {
    "risk_aligned": "Offer effect grows with churn risk: the riskiest customers are also the most persuadable.",
    "realistic": "Service-quality churners are 'lost causes' a discount can't save; loyal customers are "
                 "'sleeping dogs' an offer can wake up.",
}
SERVING_SCENARIO = "realistic"


def true_relative_effect(X: pd.DataFrame, scenario: str = SERVING_SCENARIO) -> np.ndarray:
    """Relative churn reduction caused by a 20%-off-for-3-months offer (ground truth)."""
    r = np.full(len(X), 0.05)
    r += 0.30 * (X["Contract"] == "Month-to-month")         # no lock-in -> offer persuades
    r += 0.15 * (X["MonthlyCharges"] > 70)                   # price-sensitive
    r += 0.10 * (X["tenure"] <= 12)                          # still forming the habit
    r = r.to_numpy(copy=True)
    # Leaving because of bad service (fiber without tech support), not price.
    service_issue = ((X["InternetService"] == "Fiber optic") & (X["TechSupport"] == "No")).to_numpy()
    # "Sleeping dogs": loyal long-contract customers reminded to shop around.
    sleeping = ((X["Contract"] == "Two year") & (X["tenure"] > 48)).to_numpy()
    if scenario == "risk_aligned":
        r -= 0.25 * service_issue          # discount helps them a little less
    elif scenario == "realistic":
        r = np.where(service_issue, 0.0, r)  # discount does nothing: lost causes
    else:
        raise ValueError(f"unknown scenario {scenario!r}")
    return np.where(sleeping, -0.60, r)


def simulate_experiment(df: pd.DataFrame, p0: np.ndarray, seed=RANDOM_STATE,
                        scenario: str = SERVING_SCENARIO) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = df[FEATURES]
    p1 = np.clip(p0 * (1 - true_relative_effect(X, scenario)), 0, 1)
    t = rng.binomial(1, 0.5, len(df))
    y = rng.binomial(1, np.where(t == 1, p1, p0))
    out = df[["customerID"] + FEATURES].copy()
    out["treatment"], out["churn"] = t, y
    out["p0_true"], out["p1_true"], out["uplift_true"] = p0, p1, p0 - p1
    return out


# --------------------------------------------------------------------------- evaluation
def qini_curve(uplift_pred, t, y):
    """Qini curve on observed outcomes (retention = 1 - churn). Returns (fraction, qini)."""
    order = np.argsort(-uplift_pred)
    t, retained = t[order], 1 - y[order]
    n_t, n_c = np.cumsum(t), np.cumsum(1 - t)
    r_t, r_c = np.cumsum(retained * t), np.cumsum(retained * (1 - t))
    qini = r_t - r_c * np.divide(n_t, n_c, out=np.zeros_like(n_t, dtype=float), where=n_c > 0)
    frac = np.arange(1, len(t) + 1) / len(t)
    return np.r_[0, frac], np.r_[0, qini]


def qini_coefficient(uplift_pred, t, y) -> float:
    frac, q = qini_curve(uplift_pred, t, y)
    random_line = frac * q[-1]
    return float(np.trapezoid(q - random_line, frac) / len(t))


def offer_economics(X: pd.DataFrame, p1: np.ndarray, uplift: np.ndarray) -> np.ndarray:
    """Expected net value ($) of sending the offer to each customer."""
    monthly = X["MonthlyCharges"].to_numpy()
    value_saved = uplift * monthly * CLV_MONTHS
    discount_cost = OFFER_DISCOUNT * OFFER_MONTHS * monthly * (1 - p1)  # paid by customers who stay
    return value_saved - discount_cost


def evaluate_policies(test: pd.DataFrame, scores: dict[str, np.ndarray]) -> pd.DataFrame:
    """Value each targeting policy against the known ground truth."""
    X = test[FEATURES]
    true_value = offer_economics(X, test["p1_true"].to_numpy(), test["uplift_true"].to_numpy())
    budget = int(0.2 * len(test))
    rows = []

    def record(name, mask):
        rows.append({
            "policy": name,
            "customers_targeted": int(mask.sum()),
            "churners_prevented": round(float(test["uplift_true"].to_numpy()[mask].sum()), 1),
            "net_value_usd": round(float(true_value[mask].sum()), 0),
            "discount_cost_usd": round(float((OFFER_DISCOUNT * OFFER_MONTHS * X["MonthlyCharges"].to_numpy()
                                              * (1 - test["p1_true"].to_numpy()))[mask].sum()), 0),
        })

    n = len(test)
    record("No offers", np.zeros(n, bool))
    record("Offer to everyone", np.ones(n, bool))
    top = lambda s, k=budget: np.isin(np.arange(n), np.argsort(-s)[:k])
    uplift_mask = scores["expected_value"] > 0
    record("Top 20% by churn risk (traditional)", top(scores["churn_risk"]))
    record("Top 20% by predicted uplift", top(scores["uplift"]))
    record("Churn risk, same # of offers as uplift policy", top(scores["churn_risk"], int(uplift_mask.sum())))
    record("Uplift model: offer if expected value > 0", uplift_mask)
    record("Oracle (true effect, upper bound)", true_value > 0)
    return pd.DataFrame(rows)


def plot_qini(curves: dict, fname="uplift_qini.png"):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, (frac, q) in curves.items():
        ax.plot(frac * 100, q, label=name, lw=2 if name != "Random" else 1,
                ls="--" if name == "Random" else "-")
    ax.set(xlabel="% of customers targeted (sorted by score)", ylabel="Incremental customers retained",
           title="Qini curve (held-out experiment data)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / fname)
    plt.close(fig)


def plot_policies(policies: pd.DataFrame):
    p = policies[policies["policy"] != "No offers"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    colors = ["#d1495b" if v < 0 else "#3b6fb6" for v in p["net_value_usd"]]
    ax.barh(p["policy"], p["net_value_usd"], color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.invert_yaxis()
    ax.set(xlabel="Net value vs. no offers ($, test set)", title="Which customers should get the offer?")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "uplift_policy_value.png")
    plt.close(fig)


def plot_segment_uplift(test, pred_uplift):
    seg = test.assign(pred=pred_uplift).groupby("Contract")[["uplift_true", "pred"]].mean()
    fig, ax = plt.subplots(figsize=(6, 3.4))
    x = np.arange(len(seg))
    ax.bar(x - 0.2, seg["uplift_true"] * 100, 0.4, label="True effect", color="#9aa5b1")
    ax.bar(x + 0.2, seg["pred"] * 100, 0.4, label="Model estimate", color="#3b6fb6")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, seg.index)
    ax.set(ylabel="Churn reduction (pp)", title="Offer effect by contract type")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "uplift_by_contract.png")
    plt.close(fig)


def plot_scenarios(summary: dict):
    names = ["Top 20% by churn risk (traditional)", "Churn risk, same # of offers as uplift policy",
             "Uplift model: offer if expected value > 0", "Oracle (true effect, upper bound)"]
    labels = ["Risk top 20%", "Risk, same budget", "Uplift model", "Oracle"]
    colors = ["#9aa5b1", "#6c7a89", "#3b6fb6", "#2a9d8f"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(summary))
    w = 0.2
    for j, (n, lab, c) in enumerate(zip(names, labels, colors)):
        vals = [summary[sc]["policy_value"][n] / 1000 for sc in summary]
        ax.bar(x + (j - 1.5) * w, vals, w, label=lab, color=c)
    ax.set_xticks(x, [sc.replace("_", "-") for sc in summary])
    ax.set(ylabel="Net value ($k, test set)", title="When does uplift modelling beat risk targeting?")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "uplift_scenarios.png")
    plt.close(fig)


# --------------------------------------------------------------------------- main
def run_scenario(df, p0, churn_pipe, scenario: str, make_plots: bool) -> dict:
    print(f"\n--- scenario: {scenario} ---  {SCENARIOS[scenario]}")
    exp = simulate_experiment(df, p0, scenario=scenario)
    exp.to_csv(PROCESSED_DIR / f"simulated_experiment_{scenario}.csv", index=False)
    print(f"Experiment: {exp.treatment.mean():.0%} treated | churn control "
          f"{exp.loc[exp.treatment == 0, 'churn'].mean():.1%} vs treated "
          f"{exp.loc[exp.treatment == 1, 'churn'].mean():.1%}")

    strata = exp["treatment"].astype(str) + exp["churn"].astype(str)
    train, test = train_test_split(exp, test_size=0.3, stratify=strata, random_state=RANDOM_STATE)
    Xtr, ttr, ytr = train[FEATURES], train["treatment"].to_numpy(), train["churn"].to_numpy()
    Xte, tte, yte = test[FEATURES], test["treatment"].to_numpy(), test["churn"].to_numpy()

    results, curves, learners = {}, {}, {}
    for cls in (SLearner, TLearner, XLearner):
        m = cls().fit(Xtr, ttr, ytr)
        u = m.predict_uplift(Xte)
        results[m.name] = {
            "qini_coefficient": round(qini_coefficient(u, tte, yte), 4),
            "corr_with_true_uplift": round(float(np.corrcoef(u, test["uplift_true"])[0, 1]), 4),
            "pehe": round(float(np.sqrt(np.mean((u - test["uplift_true"]) ** 2))), 4),
        }
        curves[m.name] = qini_curve(u, tte, yte)
        learners[m.name] = m
        print(f"  {m.name}: {results[m.name]}")

    churn_risk = churn_pipe.predict_proba(Xte)[:, 1]
    results["churn-risk ranking (non-causal)"] = {
        "qini_coefficient": round(qini_coefficient(churn_risk, tte, yte), 4),
        "corr_with_true_uplift": round(float(np.corrcoef(churn_risk, test["uplift_true"])[0, 1]), 4),
    }
    curves["Churn risk (non-causal)"] = qini_curve(churn_risk, tte, yte)
    frac, q = next(iter(curves.values()))
    curves["Random"] = (frac, frac * q[-1])

    best_name = max(learners, key=lambda k: results[k]["qini_coefficient"])
    best = learners[best_name]
    print(f"Selected uplift learner: {best_name}")

    _, p1_hat = best.predict_outcomes(Xte)
    u_hat = best.predict_uplift(Xte)
    ev = offer_economics(Xte, p1_hat, u_hat)
    policies = evaluate_policies(test, {"churn_risk": churn_risk, "uplift": u_hat, "expected_value": ev})
    print(policies.to_string(index=False))

    if make_plots:
        plot_qini(curves)
        plot_policies(policies)
        plot_segment_uplift(test, u_hat)

    pol = policies.set_index("policy")["net_value_usd"]
    return {
        "description": SCENARIOS[scenario],
        "selected_learner": best_name,
        "learners": results,
        "policies": policies.to_dict(orient="records"),
        "policy_value": pol.to_dict(),
        "headline": {
            "uplift_policy_net_value": float(pol["Uplift model: offer if expected value > 0"]),
            "churn_risk_top20_net_value": float(pol["Top 20% by churn risk (traditional)"]),
            "churn_risk_same_budget_net_value": float(pol["Churn risk, same # of offers as uplift policy"]),
            "offer_everyone_net_value": float(pol["Offer to everyone"]),
            "oracle_net_value": float(pol["Oracle (true effect, upper bound)"]),
        },
        "_learner_cls": type(best),
        "_exp": exp,
    }


def main():
    df = load_dataset()
    churn_art = joblib.load(CHURN_MODEL_PATH)
    from .train import make_xgb  # same tuned architecture, refit out-of-fold

    print("Computing leakage-free baseline churn probabilities (5-fold OOF)...")
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    p0 = cross_val_predict(make_xgb(**churn_art["xgb_params"]), df[FEATURES], df[TARGET], cv=cv,
                           method="predict_proba")[:, 1]

    summary = {sc: run_scenario(df, p0, churn_art["pipeline"], sc, make_plots=sc == SERVING_SCENARIO)
               for sc in SCENARIOS}
    plot_scenarios(summary)

    # Retrain the selected learner on all experiment data of the serving scenario.
    serving = summary[SERVING_SCENARIO]
    exp = serving.pop("_exp")
    final = serving.pop("_learner_cls")().fit(exp[FEATURES], exp["treatment"].to_numpy(), exp["churn"].to_numpy())
    joblib.dump({"learner": final, "name": serving["selected_learner"], "scenario": SERVING_SCENARIO},
                UPLIFT_MODEL_PATH)
    for sc in summary.values():
        sc.pop("_exp", None)
        sc.pop("_learner_cls", None)

    report = {
        "serving_scenario": SERVING_SCENARIO,
        **{k: serving[k] for k in ("selected_learner", "learners", "policies", "headline")},
        "scenarios": summary,
        "assumptions": {"discount": OFFER_DISCOUNT, "offer_months": OFFER_MONTHS, "clv_months": CLV_MONTHS,
                        "test_customers": int(round(0.3 * len(df)))},
    }
    (REPORTS_DIR / "uplift_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, sc in summary.items():
        print(f"{name}: {json.dumps(sc['headline'])}")


if __name__ == "__main__":
    main()
