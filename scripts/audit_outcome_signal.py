#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from firac_appeals.utils.io import read_jsonl


OUTCOME_PATTERNS = {
    "explicit_allowed": r"\b(appeal|petition|application|writ)\s+(?:was\s+|is\s+|has been\s+)?(?:partly\s+)?allowed\b",
    "explicit_dismissed": r"\b(appeal|petition|application|writ)\s+(?:was\s+|is\s+|has been\s+)?dismissed\b",
    "result_set_aside": r"\b(?:order|judgment|conviction|sentence)\s+(?:was\s+|is\s+)?set aside\b",
    "result_upheld": r"\b(?:order|judgment|conviction|sentence)\s+(?:was\s+|is\s+)?upheld\b",
    "court_disposition": r"\b(?:court|bench)\s+(?:therefore\s+|ultimately\s+)?(?:allowed|dismissed|rejected|granted)\b",
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def load(path: str) -> list[dict[str, str]]:
    output = []
    for row in read_jsonl(path):
        output.append(
            {
                "case_id": str(row["case_id"]),
                "label": str(row["target"]["label"]).lower(),
                "text": str(row.get("input", {}).get("input_text", "")),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit lexical outcome signal and exact overlap.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-features", type=int, default=100_000)
    args = parser.parse_args()

    train = load(args.train)
    test = load(args.test)
    train_text = [row["text"] for row in train]
    test_text = [row["text"] for row in test]
    train_labels = [row["label"] for row in train]
    test_labels = [row["label"] for row in test]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.995,
        max_features=args.max_features,
        sublinear_tf=True,
    )
    train_matrix = vectorizer.fit_transform(train_text)
    test_matrix = vectorizer.transform(test_text)
    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=20260604,
    )
    classifier.fit(train_matrix, train_labels)
    predicted = classifier.predict(test_matrix)

    features = np.asarray(vectorizer.get_feature_names_out())
    coefficients = classifier.coef_[0]
    negative_label, positive_label = classifier.classes_
    top_count = min(30, len(features))

    train_hashes: dict[str, list[dict[str, str]]] = {}
    for row in train:
        digest = hashlib.sha256(normalize(row["text"]).encode("utf-8")).hexdigest()
        train_hashes.setdefault(digest, []).append(row)
    exact_matches = []
    for row in test:
        digest = hashlib.sha256(normalize(row["text"]).encode("utf-8")).hexdigest()
        for matched in train_hashes.get(digest, []):
            exact_matches.append(
                {
                    "test_case_id": row["case_id"],
                    "train_case_id": matched["case_id"],
                    "test_label": row["label"],
                    "train_label": matched["label"],
                }
            )

    pattern_counts = {}
    for name, pattern in OUTCOME_PATTERNS.items():
        compiled = re.compile(pattern, re.IGNORECASE)
        pattern_counts[name] = {
            split: dict(Counter(row["label"] for row in rows if compiled.search(row["text"])))
            for split, rows in (("train", train), ("test", test))
        }

    payload = {
        "train": args.train,
        "test": args.test,
        "train_count": len(train),
        "test_count": len(test),
        "train_labels": dict(Counter(train_labels)),
        "test_labels": dict(Counter(test_labels)),
        "case_id_overlap_count": len(set(row["case_id"] for row in train) & set(row["case_id"] for row in test)),
        "exact_normalized_text_matches": exact_matches,
        "explicit_outcome_patterns": pattern_counts,
        "tfidf_logistic_regression": {
            "accuracy": float(accuracy_score(test_labels, predicted)),
            "macro_f1": float(f1_score(test_labels, predicted, average="macro")),
            "mcc": float(matthews_corrcoef(test_labels, predicted)),
            "prediction_distribution": dict(Counter(predicted)),
            "class_order": [str(value) for value in classifier.classes_],
            f"top_features_for_{positive_label}": [
                {"feature": str(features[index]), "coefficient": float(coefficients[index])}
                for index in np.argsort(coefficients)[-top_count:][::-1]
            ],
            f"top_features_for_{negative_label}": [
                {"feature": str(features[index]), "coefficient": float(coefficients[index])}
                for index in np.argsort(coefficients)[:top_count]
            ],
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
