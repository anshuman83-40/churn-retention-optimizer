"""Run the full pipeline end to end: EDA -> churn model -> uplift model."""
from . import eda, train, uplift


def main():
    for step in (eda, train, uplift):
        print(f"\n===== {step.__name__} =====")
        step.main()


if __name__ == "__main__":
    main()
