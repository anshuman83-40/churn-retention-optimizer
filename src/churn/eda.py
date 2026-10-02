"""Exploratory data analysis: saves figures and a markdown summary to reports/."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import FIGURES_DIR, REPORTS_DIR, TARGET
from .data import load_dataset

plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED = "#3b6fb6", "#d1495b"


def churn_rate_by(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby(col, observed=True)[TARGET].mean().sort_values(ascending=False)


def bar(rates: pd.Series, title: str, fname: str, overall: float):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(rates.index.astype(str), rates.values * 100, color=BLUE)
    ax.axhline(overall * 100, color=RED, ls="--", lw=1, label=f"overall {overall:.0%}")
    ax.set_ylabel("Churn rate (%)")
    ax.set_title(title)
    ax.legend(frameon=False)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / fname)
    plt.close(fig)


def main():
    df = load_dataset()
    overall = df[TARGET].mean()
    lines = [
        "# EDA summary",
        "",
        f"- Customers: **{len(df):,}**, features: **{df.shape[1] - 2}**",
        f"- Overall churn rate: **{overall:.1%}** (class imbalance ~{(1 - overall) / overall:.1f}:1)",
        "",
        "## Churn rate by key segments",
        "",
    ]

    for col, title in [
        ("Contract", "Churn by contract type"),
        ("tenure_group", "Churn by tenure"),
        ("PaymentMethod", "Churn by payment method"),
        ("InternetService", "Churn by internet service"),
        ("TechSupport", "Churn by tech support"),
    ]:
        rates = churn_rate_by(df, col)
        bar(rates, title, f"eda_{col.lower()}.png", overall)
        top, low = rates.index[0], rates.index[-1]
        lines.append(
            f"- **{col}**: highest `{top}` ({rates.iloc[0]:.0%}), lowest `{low}` ({rates.iloc[-1]:.0%}) "
            f"-> {rates.iloc[0] / max(rates.iloc[-1], 1e-9):.1f}x difference"
        )

    # Monthly charges distribution by churn
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for label, color in [(0, BLUE), (1, RED)]:
        ax.hist(df.loc[df[TARGET] == label, "MonthlyCharges"], bins=30, alpha=0.6, color=color,
                label="Churned" if label else "Stayed", density=True)
    ax.set_xlabel("Monthly charges ($)")
    ax.set_title("Monthly charges: churned vs stayed")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_monthly_charges.png")
    plt.close(fig)

    med = df.groupby(TARGET)[["tenure", "MonthlyCharges", "num_services"]].median()
    lines += [
        "",
        "## Medians: stayed vs churned",
        "",
        med.rename(index={0: "Stayed", 1: "Churned"}).to_markdown(),
        "",
        "## Takeaways",
        "",
        "1. Month-to-month contracts are the single biggest churn driver; long contracts barely churn.",
        "2. Churn is front-loaded: the first 6-12 months are the danger zone.",
        "3. Electronic-check payers and fiber-optic customers churn far more than average.",
        "4. Customers without tech support / online security churn more -> service bundles retain.",
        "5. Churners pay *higher* monthly bills on average -> price sensitivity matters.",
    ]
    (REPORTS_DIR / "eda_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
