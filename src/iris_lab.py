from pathlib import Path
import argparse
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


def run(output_dir='reports'):
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.25, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(confusion_matrix(y_test, pred)).to_csv(out / 'iris_confusion_matrix.csv', index=False)
    pd.DataFrame([{'accuracy': accuracy_score(y_test, pred)}]).to_csv(out / 'iris_metrics.csv', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='reports')
    args = parser.parse_args()
    run(args.output_dir)
    print(args.output_dir)


if __name__ == '__main__':
    main()
