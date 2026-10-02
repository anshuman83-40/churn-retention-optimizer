import numpy as np
import pandas as pd

from churn.config import FEATURES
from churn.data import load_dataset
from churn.learners import SLearner, TLearner
from churn.uplift import qini_coefficient, simulate_experiment, true_relative_effect


def test_qini_perfect_ranking_beats_reversed():
    rng = np.random.default_rng(0)
    n = 2000
    tau = rng.uniform(-0.2, 0.4, n)
    t = rng.binomial(1, 0.5, n)
    y = rng.binomial(1, np.clip(0.4 - t * tau, 0, 1))
    assert qini_coefficient(tau, t, y) > 0 > qini_coefficient(-tau, t, y)


def test_sleeping_dogs_have_negative_effect():
    X = pd.DataFrame([{"Contract": "Two year", "tenure": 60, "MonthlyCharges": 50,
                       "InternetService": "DSL", "TechSupport": "Yes"}])
    assert true_relative_effect(X)[0] < 0


def test_lost_causes_unaffected_only_in_realistic_scenario():
    X = pd.DataFrame([{"Contract": "Month-to-month", "tenure": 3, "MonthlyCharges": 90,
                       "InternetService": "Fiber optic", "TechSupport": "No"}])
    assert true_relative_effect(X, "realistic")[0] == 0
    assert true_relative_effect(X, "risk_aligned")[0] > 0


def test_learners_recover_effect_direction():
    df = load_dataset().sample(3000, random_state=0)
    p0 = np.full(len(df), 0.3)
    exp = simulate_experiment(df, p0, seed=1)
    for cls in (SLearner, TLearner):
        u = cls().fit(exp[FEATURES], exp["treatment"].to_numpy(), exp["churn"].to_numpy()).predict_uplift(exp[FEATURES])
        assert np.corrcoef(u, exp["uplift_true"])[0, 1] > 0.3
