"""Train churn models: Logistic Regression baseline vs tuned XGBoost, with SHAP explanations."""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .config import (
    CHURN_MODEL_PATH,
    FEATURES,
    FIGURES_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET,
    TEST_SIZE,
)
from .data import load_dataset
from .features import aggregate_by_feature, build_preprocessor, original_feature_groups

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def best_f1_threshold(y_true, proba) -> float:
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    return float(thr[np.argmax(f1[:-1])])


def evaluate(y_true, proba, threshold) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": round(roc_auc_score(y_true, proba), 4),
        "pr_auc": round(average_precision_score(y_true, proba), 4),
        "f1": round(f1_score(y_true, pred), 4),
        "precision": round(precision_score(y_true, pred), 4),
        "recall": round(recall_score(y_true, pred), 4),
        "brier": round(brier_score_loss(y_true, proba), 4),
        "threshold": round(threshold, 4),
    }


def make_logreg() -> Pipeline:
    return Pipeline([
        ("pre", build_preprocessor()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)),
    ])


def make_xgb(**params) -> Pipeline:
    return Pipeline([
        ("pre", build_preprocessor()),
        ("clf", XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1, **params)),
    ])


def tune_xgb(X, y) -> dict:
    space = {
        "clf__n_estimators": [200, 300, 500, 800],
        "clf__max_depth": [2, 3, 4, 5],
        "clf__learning_rate": [0.01, 0.02, 0.05, 0.1],
        "clf__subsample": [0.7, 0.8, 1.0],
        "clf__colsample_bytree": [0.6, 0.8, 1.0],
        "clf__min_child_weight": [1, 3, 5, 10],
        "clf__reg_lambda": [0.5, 1, 3, 10],
    }
    search = RandomizedSearchCV(make_xgb(), space, n_iter=30, scoring="roc_auc", cv=CV,
                                random_state=RANDOM_STATE, n_jobs=-1)
    search.fit(X, y)
    print(f"  best CV ROC-AUC: {search.best_score_:.4f}")
    return {k.replace("clf__", ""): v for k, v in search.best_params_.items()}, search.best_score_


def plot_curves(y_test, probas: dict):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_test, p):.3f})")
        prec, rec, _ = precision_recall_curve(y_test, p)
        axes[1].plot(rec, prec, label=f"{name} (AP {average_precision_score(y_test, p):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    axes[1].axhline(y_test.mean(), color="k", ls="--", lw=0.8)
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve")
    for ax in axes:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_roc_pr.png")
    plt.close(fig)


def shap_report(model: Pipeline, X_test: pd.DataFrame) -> dict:
    pre, clf = model.named_steps["pre"], model.named_steps["clf"]
    Xt = pre.transform(X_test)
    names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(Xt)

    plt.figure()
    shap.summary_plot(sv, pd.DataFrame(Xt, columns=names), max_display=15, show=False)
    plt.title("SHAP: what drives churn predictions")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.png", bbox_inches="tight")
    plt.close()

    groups = original_feature_groups(pre)
    importance = aggregate_by_feature(np.abs(sv).mean(axis=0), groups)
    importance = dict(sorted(importance.items(), key=lambda kv: -kv[1]))

    top = list(importance.items())[:12][::-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh([k for k, _ in top], [v for _, v in top], color="#3b6fb6")
    ax.set(xlabel="mean |SHAP value| (log-odds)", title="Global feature importance")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "shap_importance.png")
    plt.close(fig)
    return {k: round(v, 4) for k, v in importance.items()}


def main():
    df = load_dataset()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, df["customerID"], test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    print("Baseline: Logistic Regression")
    logreg = make_logreg()
    lr_oof = cross_val_predict(logreg, X_train, y_train, cv=CV, method="predict_proba")[:, 1]
    lr_thr = best_f1_threshold(y_train, lr_oof)
    logreg.fit(X_train, y_train)
    lr_test = logreg.predict_proba(X_test)[:, 1]

    print("Tuning XGBoost (30 configs x 5 folds)...")
    params, cv_auc = tune_xgb(X_train, y_train)
    xgb = make_xgb(**params)
    xgb_oof = cross_val_predict(xgb, X_train, y_train, cv=CV, method="predict_proba")[:, 1]
    xgb_thr = best_f1_threshold(y_train, xgb_oof)
    xgb.fit(X_train, y_train)
    xgb_test = xgb.predict_proba(X_test)[:, 1]

    results = {
        "logistic_regression": evaluate(y_test, lr_test, lr_thr),
        "xgboost": evaluate(y_test, xgb_test, xgb_thr),
    }
    results["xgboost"]["cv_roc_auc"] = round(cv_auc, 4)
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model, best_thr = (xgb, xgb_thr) if best_name == "xgboost" else (logreg, lr_thr)
    print(json.dumps(results, indent=2))
    print(f"Selected model: {best_name}")

    # Lift: how concentrated is churn in the top-risk decile?
    best_test = xgb_test if best_name == "xgboost" else lr_test
    top_decile = best_test >= np.quantile(best_test, 0.9)
    lift = float(y_test[top_decile].mean() / y_test.mean())

    plot_curves(y_test, {"Logistic Regression": lr_test, "XGBoost": xgb_test})
    importance = shap_report(xgb, X_test)

    joblib.dump(
        {"pipeline": best_model, "threshold": best_thr, "model_name": best_name, "xgb_params": params},
        CHURN_MODEL_PATH,
    )
    pd.DataFrame({"customerID": id_test.values, "y_true": y_test.values, "churn_proba": best_test}).to_csv(
        PROCESSED_DIR / "test_predictions.csv", index=False
    )
    report = {
        "selected_model": best_name,
        "metrics": results,
        "top_decile_lift": round(lift, 2),
        "xgb_params": params,
        "feature_importance": importance,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    (REPORTS_DIR / "churn_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Top-decile lift: {lift:.2f}x | saved model -> {CHURN_MODEL_PATH}")


if __name__ == "__main__":
    main()
