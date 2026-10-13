"""ChurnUpliftNet: a neural network written from scratch in NumPy (no ML library).

One network answers both questions the project asks:

    shared trunk  x -> h            what kind of customer is this?
    base head     h -> z0           churn risk if we do nothing        (logit)
    effect head   h -> delta        how the retention offer shifts it  (logit change)

    P(churn | no offer) = sigmoid(z0)
    P(churn | offer)    = sigmoid(z0 + delta)
    uplift              = P(churn | no offer) - P(churn | offer)

The offer's effect is a *correction on top of* the baseline risk, and the effect head is
regularised harder than the rest, so the model only claims an effect the data supports.

Everything is implemented here by hand: feature encoding, forward pass, backpropagation,
the Adam optimiser, mini-batch training with early stopping, seed-ensembling, a Newton-method
calibration of the offer effect and the per-customer explanations (occlusion). Only NumPy and pandas are used.
"""
import numpy as np
import pandas as pd

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE


# --------------------------------------------------------------------------- feature encoding
class Encoder:
    """Standardises numeric columns and one-hot encodes categorical ones."""

    def fit(self, X: pd.DataFrame):
        num = X[NUMERIC_FEATURES].to_numpy(dtype=float)
        self.mean_ = num.mean(axis=0)
        self.std_ = num.std(axis=0) + 1e-8
        self.categories_ = {c: sorted(X[c].astype(str).unique()) for c in CATEGORICAL_FEATURES}
        # which encoded columns belong to which original feature (used for explanations)
        self.slices_, start = {}, 0
        for c in NUMERIC_FEATURES:
            self.slices_[c] = slice(start, start + 1)
            start += 1
        for c in CATEGORICAL_FEATURES:
            self.slices_[c] = slice(start, start + len(self.categories_[c]))
            start += len(self.categories_[c])
        self.n_features_ = start
        self.baseline_ = self.transform(X).mean(axis=0)  # the "average customer"
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(X), self.n_features_))
        out[:, : len(NUMERIC_FEATURES)] = (X[NUMERIC_FEATURES].to_numpy(dtype=float) - self.mean_) / self.std_
        for c in CATEGORICAL_FEATURES:
            values = X[c].astype(str).to_numpy()
            for j, cat in enumerate(self.categories_[c]):  # unseen categories stay all-zero
                out[values == cat, self.slices_[c].start + j] = 1.0
        return out


# --------------------------------------------------------------------------- building blocks
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def relu(a):
    return np.maximum(a, 0.0)


class Adam:
    """Adam optimiser (Kingma & Ba, 2015) for a dict of parameter arrays."""

    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# --------------------------------------------------------------------------- one network
TRUNK, BASE, EFFECT = ("W1", "W2"), ("Wb1", "Wb2"), ("We1", "We2")


def init_params(n_in, hidden, head, rng):
    def he(a, b):  # He initialisation, suited to ReLU layers
        return rng.normal(0.0, np.sqrt(2.0 / a), size=(a, b))

    h1, h2 = hidden
    return {
        "W1": he(n_in, h1), "b1": np.zeros(h1), "W2": he(h1, h2), "b2": np.zeros(h2),
        "Wb1": he(h2, head), "bb1": np.zeros(head), "Wb2": he(head, 1) * 0.1, "bb2": np.zeros(1),
        # the effect head starts at exactly zero output: "no effect until the data says otherwise"
        "We1": he(h2, head), "be1": np.zeros(head), "We2": np.zeros((head, 1)), "be2": np.zeros(1),
    }


def forward(p, X, t):
    """Returns (z0, delta, churn probability under the given treatment t, cache for backprop)."""
    a1 = relu(X @ p["W1"] + p["b1"])
    a2 = relu(a1 @ p["W2"] + p["b2"])
    ab = relu(a2 @ p["Wb1"] + p["bb1"])
    z0 = (ab @ p["Wb2"] + p["bb2"]).ravel()
    ae = relu(a2 @ p["We1"] + p["be1"])
    delta = (ae @ p["We2"] + p["be2"]).ravel()
    prob = sigmoid(z0 + t * delta)
    return z0, delta, prob, (X, a1, a2, ab, ae)


def loss_and_grads(p, X, t, y, l2, l2_effect):
    """Binary cross-entropy + L2 penalties, and its exact gradient by backpropagation."""
    n = len(y)
    _, _, prob, (X, a1, a2, ab, ae) = forward(p, X, t)
    eps = 1e-12
    loss = -np.mean(y * np.log(prob + eps) + (1 - y) * np.log(1 - prob + eps))
    loss += 0.5 * l2 * sum((p[k] ** 2).sum() for k in TRUNK + BASE)
    loss += 0.5 * l2_effect * sum((p[k] ** 2).sum() for k in EFFECT)

    g = {}
    dz = ((prob - y) / n)[:, None]          # d loss / d (z0 + t*delta)
    # base head
    g["Wb2"] = ab.T @ dz + l2 * p["Wb2"]
    g["bb2"] = dz.sum(axis=0)
    dab = (dz @ p["Wb2"].T) * (ab > 0)
    g["Wb1"] = a2.T @ dab + l2 * p["Wb1"]
    g["bb1"] = dab.sum(axis=0)
    # effect head: only treated rows (t = 1) send a gradient
    dd = dz * t[:, None]
    g["We2"] = ae.T @ dd + l2_effect * p["We2"]
    g["be2"] = dd.sum(axis=0)
    dae = (dd @ p["We2"].T) * (ae > 0)
    g["We1"] = a2.T @ dae + l2_effect * p["We1"]
    g["be1"] = dae.sum(axis=0)
    # shared trunk receives gradient from both heads
    da2 = (dab @ p["Wb1"].T + dae @ p["We1"].T) * (a2 > 0)
    g["W2"] = a1.T @ da2 + l2 * p["W2"]
    g["b2"] = da2.sum(axis=0)
    da1 = (da2 @ p["W2"].T) * (a1 > 0)
    g["W1"] = X.T @ da1 + l2 * p["W1"]
    g["b1"] = da1.sum(axis=0)
    return loss, g


def train_one(X, t, y, Xv, tv, yv, *, hidden, head, lr, l2, l2_effect, epochs, batch, patience, seed):
    """Mini-batch Adam with early stopping on a validation set. Returns the best parameters."""
    rng = np.random.default_rng(seed)
    p = init_params(X.shape[1], hidden, head, rng)
    opt = Adam(p, lr=lr)
    best, best_loss, wait = {k: v.copy() for k, v in p.items()}, np.inf, 0
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for i in range(0, len(y), batch):
            idx = order[i:i + batch]
            _, grads = loss_and_grads(p, X[idx], t[idx], y[idx], l2, l2_effect)
            opt.step(p, grads)
        val_loss, _ = loss_and_grads(p, Xv, tv, yv, 0.0, 0.0)
        if val_loss < best_loss - 1e-5:
            best, best_loss, wait = {k: v.copy() for k, v in p.items()}, val_loss, 0
        else:
            wait += 1
            if wait >= patience:
                break
    return best


# --------------------------------------------------------------------------- the model
class ChurnUpliftNet:
    """Churn + uplift model. An ensemble of `n_models` networks that differ only in their seed."""

    name = "ChurnUpliftNet"

    def __init__(self, hidden=(32, 16), head=8, lr=2e-3, l2=1e-3, l2_effect=1e-2, epochs=300, batch=128,
                 patience=20, n_models=7, val_fraction=0.15, seed=RANDOM_STATE):
        self.hp = dict(hidden=hidden, head=head, lr=lr, l2=l2, l2_effect=l2_effect, epochs=epochs,
                       batch=batch, patience=patience)
        self.n_models, self.val_fraction, self.seed = n_models, val_fraction, seed

    def fit(self, X: pd.DataFrame, t=None, y=None):
        """t = 1 where the customer received the offer. Omit t to train churn prediction only."""
        y = np.asarray(y, dtype=float)
        t = np.zeros(len(y)) if t is None else np.asarray(t, dtype=float)
        self.encoder = Encoder().fit(X)
        Xe = self.encoder.transform(X)
        self.models = []
        for m in range(self.n_models):
            rng = np.random.default_rng(self.seed + 1000 * m)
            order = rng.permutation(len(y))  # each member gets its own train/validation split
            n_val = int(len(y) * self.val_fraction)
            va, tr = order[:n_val], order[n_val:]
            self.models.append(train_one(Xe[tr], t[tr], y[tr], Xe[va], t[va], y[va], seed=self.seed + m, **self.hp))
        self.effect_shift, self.effect_scale = 0.0, 1.0
        if 0 < t.sum() < len(t):
            self._calibrate_effect(Xe[t == 1], y[t == 1])
        return self

    def _calibrate_effect(self, Xe_treated, y_treated, steps=25):
        """The L2 penalty shrinks the effect head toward zero. Undo that bias with two numbers:
        treated logit = z0 + shift + scale * delta, fitted on treated customers by Newton's method
        (a 2-parameter logistic regression with z0 as a fixed offset)."""
        z0, delta = self._raw_logits(Xe_treated)
        F = np.column_stack([np.ones_like(delta), delta])
        w = np.array([0.0, 1.0])
        for _ in range(steps):
            prob = sigmoid(z0 + F @ w)
            grad = F.T @ (prob - y_treated)
            hess = (F * (prob * (1 - prob))[:, None]).T @ F + 1e-6 * np.eye(2)
            step = np.linalg.solve(hess, grad)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        self.effect_shift, self.effect_scale = float(w[0]), float(w[1])

    def _raw_logits(self, Xe):
        outs = [forward(p, Xe, np.zeros(len(Xe))) for p in self.models]
        return np.mean([o[0] for o in outs], axis=0), np.mean([o[1] for o in outs], axis=0)

    def _logits(self, Xe):
        z0, delta = self._raw_logits(Xe)
        return z0, self.effect_shift + self.effect_scale * delta

    def predict_outcomes(self, X):
        """Returns (P(churn | no offer), P(churn | offer))."""
        z0, delta = self._logits(self.encoder.transform(X))
        return sigmoid(z0), sigmoid(z0 + delta)

    def predict_uplift(self, X):
        p0, p1 = self.predict_outcomes(X)
        return p0 - p1

    def predict_proba(self, X):
        """scikit-learn style: column 1 is the churn probability without an offer."""
        p0, _ = self.predict_outcomes(X)
        return np.column_stack([1 - p0, p0])

    def explain(self, X) -> list[dict[str, float]]:
        """Per-customer risk drivers by occlusion: how much does the churn logit drop when one
        feature is replaced by the average customer's value? Positive = pushes toward churn."""
        Xe = self.encoder.transform(X)
        z_full, _ = self._logits(Xe)
        out = [{} for _ in range(len(Xe))]
        for feat, sl in self.encoder.slices_.items():
            Xo = Xe.copy()
            Xo[:, sl] = self.encoder.baseline_[sl]
            z_without, _ = self._logits(Xo)
            for i, v in enumerate(z_full - z_without):
                out[i][feat] = float(v)
        return out
