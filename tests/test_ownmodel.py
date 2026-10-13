"""Tests for ChurnUpliftNet, the from-scratch NumPy model."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from churn import ownmodel as om
from churn.config import FEATURES, TARGET
from churn.data import load_dataset


def test_backprop_matches_numerical_gradient():
    """The hand-written backpropagation must agree with finite differences on every parameter."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))
    t = rng.integers(0, 2, 40).astype(float)
    y = rng.integers(0, 2, 40).astype(float)
    p = om.init_params(6, (5, 4), 3, rng)
    p["We2"] = rng.normal(0, 0.5, p["We2"].shape)  # give the effect head a non-zero output
    _, grads = om.loss_and_grads(p, X, t, y, 1e-2, 3e-2)
    worst, h = 0.0, 1e-6
    for k in p:
        for i in np.ndindex(p[k].shape):
            old = p[k][i]
            p[k][i] = old + h
            up, _ = om.loss_and_grads(p, X, t, y, 1e-2, 3e-2)
            p[k][i] = old - h
            down, _ = om.loss_and_grads(p, X, t, y, 1e-2, 3e-2)
            p[k][i] = old
            numeric = (up - down) / (2 * h)
            worst = max(worst, abs(numeric - grads[k][i]) / max(1e-8, abs(numeric) + abs(grads[k][i])))
    assert worst < 1e-5


def test_encoder_one_hot_and_scaling():
    X = load_dataset()[FEATURES].head(500)
    enc = om.Encoder().fit(X)
    Z = enc.transform(X)
    assert Z.shape == (500, enc.n_features_)
    assert np.allclose(Z[:, :6].mean(axis=0), 0, atol=1e-9)           # numeric columns standardised
    contract = enc.slices_["Contract"]
    assert np.allclose(Z[:, contract].sum(axis=1), 1)                 # exactly one category per row
    unseen = X.head(1).assign(Contract="Weekly")
    assert enc.transform(unseen)[0, contract].sum() == 0               # unseen category -> all zeros


@pytest.fixture(scope="module")
def small_fit():
    df = load_dataset().sample(2500, random_state=1)
    train, test = df.iloc[:2000], df.iloc[2000:]
    net = om.ChurnUpliftNet(n_models=2, epochs=60).fit(train[FEATURES], y=train[TARGET])
    return net, test


def test_learns_churn(small_fit):
    net, test = small_fit
    assert roc_auc_score(test[TARGET], net.predict_proba(test[FEATURES])[:, 1]) > 0.80


def test_no_treatment_means_no_uplift(small_fit):
    """Trained without any treated customers, the effect head must stay at exactly zero."""
    net, test = small_fit
    assert np.allclose(net.predict_uplift(test[FEATURES]), 0)


def test_explanations_cover_every_feature(small_fit):
    net, test = small_fit
    contrib = net.explain(test[FEATURES].head(5))
    assert len(contrib) == 5 and set(contrib[0]) == set(FEATURES)


def test_recovers_a_planted_offer_effect():
    """Plant an offer that helps month-to-month customers only; the model must rank them highest."""
    df = load_dataset().sample(3000, random_state=2)
    rng = np.random.default_rng(3)
    t = rng.integers(0, 2, len(df))
    m2m = (df["Contract"] == "Month-to-month").to_numpy()
    y = rng.binomial(1, np.where(m2m, 0.5, 0.1) * np.where((t == 1) & m2m, 0.4, 1.0))
    net = om.ChurnUpliftNet(n_models=2, epochs=80).fit(df[FEATURES], t, y)
    u = net.predict_uplift(df[FEATURES])
    assert u[m2m].mean() > u[~m2m].mean() + 0.1
