"""Preprocessing pipeline + helpers to map one-hot columns back to original features."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )


def original_feature_groups(pre: ColumnTransformer) -> list[str]:
    """For each transformed column, the original feature it came from."""
    groups = list(NUMERIC_FEATURES)
    enc = pre.named_transformers_["cat"]
    for feat, cats in zip(CATEGORICAL_FEATURES, enc.categories_):
        groups += [feat] * len(cats)
    return groups


def aggregate_by_feature(values: np.ndarray, groups: list[str]) -> dict[str, float]:
    """Sum per-column values (e.g. SHAP) into per-original-feature values."""
    out: dict[str, float] = {}
    for v, g in zip(values, groups):
        out[g] = out.get(g, 0.0) + float(v)
    return out
