#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from firac_appeals.evaluation.downstream import normalize_rule_ref, rule_items, text_items
from firac_appeals.utils.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Dense case-retrieval baselines for FIRAC tasks.")
    parser.add_argument("--task", required=True, choices=["issue_generation", "rule_recommendation"])
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--retrieved-cases", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--cache-dir", default="runs/evaluations/downstream/dense_retrieval/cache")
    args = parser.parse_args()

    train = read_jsonl(args.train)
    eval_rows = read_jsonl(args.eval)
    train_vectors, eval_vectors = encode_rows(
        train,
        eval_rows,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        cache_dir=Path(args.cache_dir),
        cache_key=cache_key(args.task, args.train, args.eval, args.model),
    )
    neighbor_count = min(args.retrieved_cases, len(train))
    predictions: list[dict[str, Any]] = []
    for start in range(0, len(eval_rows), 128):
        query_vectors = eval_vectors[start : start + 128]
        similarities = query_vectors @ train_vectors.T
        neighbor_ids = np.argpartition(similarities, -neighbor_count, axis=1)[:, -neighbor_count:]
        for offset, row in enumerate(eval_rows[start : start + 128]):
            ids = neighbor_ids[offset]
            ids = ids[np.argsort(similarities[offset, ids])[::-1]]
            ranked = [(train[int(idx)], float(similarities[offset, idx])) for idx in ids]
            if args.task == "issue_generation":
                prediction = predict_issues(row, ranked)
            else:
                prediction = predict_rules(row, ranked, args.top_k)
            predictions.append(prediction)
    write_jsonl(args.output, predictions)
    print(json.dumps({"task": args.task, "train": len(train), "eval": len(eval_rows), "output": args.output}))


def encode_rows(
    train: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    *,
    model_name: str,
    device: str,
    batch_size: int,
    cache_dir: Path,
    cache_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    train_path = cache_dir / f"{cache_key}.train.npy"
    eval_path = cache_dir / f"{cache_key}.eval.npy"
    if train_path.exists() and eval_path.exists():
        return np.load(train_path), np.load(eval_path)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    train_vectors = model.encode(
        [input_text(row) for row in train],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    eval_vectors = model.encode(
        [input_text(row) for row in eval_rows],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(train_path, train_vectors)
    np.save(eval_path, eval_vectors)
    return train_vectors, eval_vectors


def predict_issues(row: dict[str, Any], ranked: list[tuple[dict[str, Any], float]]) -> dict[str, Any]:
    nearest = ranked[0][0] if ranked else {}
    return {
        "case_id": row.get("case_id", ""),
        "task": "issue_generation",
        "baseline": "dense_nearest_case_mpnet",
        "issues": text_items((nearest.get("target") or {}).get("issues")),
        "retrieved_case_id": nearest.get("case_id", ""),
    }


def predict_rules(
    row: dict[str, Any],
    ranked: list[tuple[dict[str, Any], float]],
    top_k: int,
) -> dict[str, Any]:
    scores: dict[str, float] = defaultdict(float)
    representatives: dict[str, dict[str, Any]] = {}
    for rank, (neighbor, similarity) in enumerate(ranked, start=1):
        for rule_rank, rule in enumerate(rule_items((neighbor.get("target") or {}).get("rules")), start=1):
            key = normalize_rule_ref(rule)
            if not key:
                continue
            scores[key] += max(similarity, 0.0) / (rank * (1.0 + 0.05 * (rule_rank - 1)))
            representatives.setdefault(key, rule)
    ordered = sorted(scores, key=lambda key: (-scores[key], key))[:top_k]
    return {
        "case_id": row.get("case_id", ""),
        "task": "rule_recommendation",
        "baseline": "dense_case_authority_retrieval_mpnet",
        "rules": [representatives[key] for key in ordered],
    }


def input_text(row: dict[str, Any]) -> str:
    return str((row.get("input") or {}).get("input_text", ""))


def cache_key(task: str, train: str, eval_path: str, model: str) -> str:
    import hashlib

    value = "|".join([task, str(Path(train).resolve()), str(Path(eval_path).resolve()), model])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


if __name__ == "__main__":
    main()
