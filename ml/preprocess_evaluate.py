from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


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

# Columns where zero is medically impossible and should be treated as missing.
# Pregnancies, DiabetesPedigreeFunction, and Age are excluded intentionally:
# - Pregnancies: 0 is a valid value
# - DiabetesPedigreeFunction and Age: contain no zeros
IMPUTE_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

MODELS = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": lambda: DecisionTreeClassifier(random_state=42),
    "KNN": lambda: KNeighborsClassifier(n_neighbors=5),
    "SVM": lambda: SVC(probability=True, random_state=42),
}


def impute_zeros(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """Replace zeros in medically impossible columns with the training median.
    Median is computed only from training data to prevent data leakage."""
    X_train = X_train.copy()
    X_test = X_test.copy()

    for col in IMPUTE_COLUMNS:
        if col not in X_train.columns:
            continue
        median = X_train[col].replace(0, pd.NA).median()
        X_train[col] = X_train[col].replace(0, median)
        X_test[col] = X_test[col].replace(0, median)

    return X_train, X_test


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    }


def run_evaluation(X_train, X_test, y_train, y_test, label: str):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}
    for name, model_fn in MODELS.items():
        model = model_fn()
        model.fit(X_train_scaled, y_train)
        results[name] = evaluate(model, X_test_scaled, y_test)

    df = pd.DataFrame(results).T.sort_values("ROC-AUC", ascending=False)

    print(f"\n=== {label} ===\n")
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))
    print()

    return df


def print_zero_summary(df: pd.DataFrame):
    print("\n=== Zero value counts per column (raw dataset) ===\n")
    zero_counts = df[FEATURE_COLUMNS].eq(0).sum()
    pct = (zero_counts / len(df) * 100).round(1)
    summary = pd.DataFrame({"Zero count": zero_counts, "Percentage": pct})
    summary = summary[summary["Zero count"] > 0]
    print(summary.to_string())
    print(f"\nTotal rows: {len(df)}")
    print(
        f"Columns with medically unrealistic zeros (will be imputed): {IMPUTE_COLUMNS}"
    )
    print()


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    print_zero_summary(df)

    X = df[FEATURE_COLUMNS]
    y = df["Outcome"]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    before = run_evaluation(
        X_train_raw,
        X_test_raw,
        y_train,
        y_test,
        label="Before preprocessing (raw zeros)",
    )

    X_train_clean, X_test_clean = impute_zeros(X_train_raw, X_test_raw)

    after = run_evaluation(
        X_train_clean,
        X_test_clean,
        y_train,
        y_test,
        label="After preprocessing (median imputation for zero values)",
    )

    print("=== ROC-AUC delta (after - before) ===\n")
    delta = (after["ROC-AUC"] - before["ROC-AUC"]).sort_values(ascending=False)
    for model, diff in delta.items():
        sign = "+" if diff >= 0 else ""
        print(f"  {model:<25} {sign}{diff:.4f}")
    print()


if __name__ == "__main__":
    main()
