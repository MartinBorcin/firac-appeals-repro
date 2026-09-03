#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from firac_appeals.utils.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Word/character TF-IDF logistic LJP baseline.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260604)
    args = parser.parse_args()

    train = read_jsonl(args.train)
    test = read_jsonl(args.test)
    train_text = [text(row) for row in train]
    test_text = [text(row) for row in test]
    train_y = np.asarray([label(row) for row in train])
    test_y = np.asarray([label(row) for row in test])

    word = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2), min_df=2, max_df=0.98,
        max_features=150_000, sublinear_tf=True,
    )
    char = TfidfVectorizer(
        analyzer="char_wb", lowercase=True, ngram_range=(3, 5), min_df=3,
        max_features=150_000, sublinear_tf=True,
    )
    train_x = hstack([word.fit_transform(train_text), char.fit_transform(train_text)], format="csr")
    test_x = hstack([word.transform(test_text), char.transform(test_text)], format="csr")
    model = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=1000, solver="liblinear",
        random_state=args.seed,
    )
    model.fit(train_x, train_y)
    predicted = model.predict(test_x)
    probabilities = model.predict_proba(test_x)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions = []
    for row, prediction, probs in zip(test, predicted, probabilities):
        predictions.append({
            "case_id": row["case_id"], "task": "outcome_prediction",
            "model": "tfidf_word_char_logistic_regression", "prediction_status": "ok",
            "label": str(prediction),
            "probabilities": {str(name): float(value) for name, value in zip(model.classes_, probs)},
        })
    write_jsonl(output / "predictions.jsonl", predictions)
    metrics = {
        "count": len(test), "train_count": len(train),
        "accuracy": float(accuracy_score(test_y, predicted)),
        "macro_f1": float(f1_score(test_y, predicted, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(test_y, predicted)),
        "prediction_distribution": {name: int(np.count_nonzero(predicted == name)) for name in model.classes_},
        "top_features": top_features(model, word, char, 30),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: metrics[key] for key in ("accuracy", "macro_f1", "mcc")}, indent=2))


def top_features(model: LogisticRegression, word: TfidfVectorizer, char: TfidfVectorizer, count: int) -> dict[str, list[dict[str, float | str]]]:
    names = np.concatenate([
        np.asarray([f"word:{value}" for value in word.get_feature_names_out()]),
        np.asarray([f"char:{value}" for value in char.get_feature_names_out()]),
    ])
    weights = model.coef_[0]
    return {
        str(model.classes_[1]): [{"feature": str(names[i]), "weight": float(weights[i])} for i in np.argsort(weights)[-count:][::-1]],
        str(model.classes_[0]): [{"feature": str(names[i]), "weight": float(weights[i])} for i in np.argsort(weights)[:count]],
    }


def text(row: dict) -> str:
    return str((row.get("input") or {}).get("input_text", ""))


def label(row: dict) -> str:
    return str((row.get("target") or {}).get("label", "")).lower()


if __name__ == "__main__":
    main()
