from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/diabetes.csv")

FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_COLUMNS]
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # --- Logistic Regression coefficients ---
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)

    coef_df = pd.DataFrame(
        {
            "Feature": FEATURE_COLUMNS,
            "Coefficient": lr.coef_[0],
            "Abs Coefficient": np.abs(lr.coef_[0]),
        }
    ).sort_values("Abs Coefficient", ascending=False)

    print("\n=== Logistic Regression Coefficients (standardized features) ===")
    print(
        "Positive = increases diabetes risk | Negative = decreases diabetes risk\n"
    )
    for _, row in coef_df.iterrows():
        direction = "+" if row["Coefficient"] >= 0 else ""
        bar = "#" * int(abs(row["Coefficient"]) * 10)
        print(
            f"  {row['Feature']:<28} {direction}{row['Coefficient']:+.4f}  {bar}"
        )

    # --- Random Forest feature importances ---
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)

    imp_df = pd.DataFrame(
        {
            "Feature": FEATURE_COLUMNS,
            "Importance": rf.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    print("\n=== Random Forest Feature Importances ===\n")
    for _, row in imp_df.iterrows():
        bar = "#" * int(row["Importance"] * 100)
        print(f"  {row['Feature']:<28} {row['Importance']:.4f}  {bar}")

    # --- Agreement between the two methods ---
    lr_rank = coef_df["Feature"].tolist()
    rf_rank = imp_df["Feature"].tolist()

    print("\n=== Feature Ranking Comparison ===\n")
    print(f"  {'Rank':<6} {'Logistic Regression':<28} {'Random Forest'}")
    print(f"  {'-'*6} {'-'*28} {'-'*28}")
    for i, (lr_f, rf_f) in enumerate(zip(lr_rank, rf_rank), start=1):
        match = "<-- agree" if lr_f == rf_f else ""
        print(f"  {i:<6} {lr_f:<28} {rf_f:<28} {match}")

    print()


if __name__ == "__main__":
    main()
