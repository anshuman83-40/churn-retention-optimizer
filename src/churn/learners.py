"""Meta-learners for uplift (heterogeneous treatment effect) estimation.

Uplift = P(churn | no offer) - P(churn | offer); positive means the offer reduces churn.
"""
import numpy as np
from xgboost import XGBClassifier, XGBRegressor

from .config import RANDOM_STATE
from .features import build_preprocessor

XGB_CLF = dict(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
               colsample_bytree=0.8, min_child_weight=5, eval_metric="logloss",
               random_state=RANDOM_STATE, n_jobs=-1)
XGB_REG = {k: v for k, v in XGB_CLF.items() if k != "eval_metric"}



class _Base:
    name = "base"

    def _fit_pre(self, X):
        self.pre = build_preprocessor().fit(X)
        return self.pre.transform(X)

    def predict_outcomes(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Returns (P(churn | no offer), P(churn | offer))."""
        raise NotImplementedError

    def predict_uplift(self, X) -> np.ndarray:
        p0, p1 = self.predict_outcomes(X)
        return p0 - p1


class SLearner(_Base):
    """One model with treatment as a feature."""
    name = "S-learner"

    def fit(self, X, t, y):
        Xt = self._fit_pre(X)
        self.m = XGBClassifier(**XGB_CLF).fit(np.column_stack([Xt, t]), y)
        return self

    def predict_outcomes(self, X):
        Xt = self.pre.transform(X)
        ones, zeros = np.ones(len(Xt)), np.zeros(len(Xt))
        return (self.m.predict_proba(np.column_stack([Xt, zeros]))[:, 1],
                self.m.predict_proba(np.column_stack([Xt, ones]))[:, 1])


class TLearner(_Base):
    """Separate churn models for control and treated customers."""
    name = "T-learner"

    def fit(self, X, t, y):
        Xt = self._fit_pre(X)
        self.m0 = XGBClassifier(**XGB_CLF).fit(Xt[t == 0], y[t == 0])
        self.m1 = XGBClassifier(**XGB_CLF).fit(Xt[t == 1], y[t == 1])
        return self

    def predict_outcomes(self, X):
        Xt = self.pre.transform(X)
        return self.m0.predict_proba(Xt)[:, 1], self.m1.predict_proba(Xt)[:, 1]


class XLearner(TLearner):
    """Kunzel et al. (2019): impute individual effects, then regress on them."""
    name = "X-learner"

    def fit(self, X, t, y, propensity=0.5):
        super().fit(X, t, y)
        Xt = self.pre.transform(X)
        # uplift = mu0 - mu1 (churn reduction)
        d_treated = self.m0.predict_proba(Xt[t == 1])[:, 1] - y[t == 1]
        d_control = y[t == 0] - self.m1.predict_proba(Xt[t == 0])[:, 1]
        self.tau1 = XGBRegressor(**XGB_REG).fit(Xt[t == 1], d_treated)
        self.tau0 = XGBRegressor(**XGB_REG).fit(Xt[t == 0], d_control)
        self.g = propensity
        return self

    def predict_uplift(self, X):
        Xt = self.pre.transform(X)
        return self.g * self.tau0.predict(Xt) + (1 - self.g) * self.tau1.predict(Xt)

    def predict_outcomes(self, X):
        p0 = self.m0.predict_proba(self.pre.transform(X))[:, 1]
        return p0, np.clip(p0 - self.predict_uplift(X), 0, 1)
