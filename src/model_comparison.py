"""Reproducible comparison of two classifiers for classroom use.

Uses the PUBLIC Iris dataset. The cross-validation runs only on training data;
the held-out test set is reserved for a final evaluation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def run(output_dir: str | Path = "reports", seed: int = 42) -> pd.DataFrame:
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.25,
        random_state=seed,
        stratify=iris.target,
    )
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)
        ),
        "k-Nearest Neighbors": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=5)
        ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    for name, model in models.items():
        scores = cross_val_score(
            model, x_train, y_train, cv=cv, scoring="accuracy"
        )
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        matrix = confusion_matrix(y_test, predictions, labels=[0, 1, 2])
        pd.DataFrame(
            matrix,
            index=[f"real_{label}" for label in iris.target_names],
            columns=[f"predito_{label}" for label in iris.target_names],
        ).to_csv(out / f"confusion_{name.lower().replace(' ', '_').replace('-', '')}.csv")
        rows.append({
            "model": name,
            "cv_accuracy_mean": round(float(scores.mean()), 4),
            "cv_accuracy_std": round(float(scores.std()), 4),
            "holdout_accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "holdout_macro_f1": round(float(f1_score(y_test, predictions, average="macro")), 4),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "model_comparison.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run(args.output_dir, args.seed)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
