#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from typing import Any

from firac_appeals.evaluation.downstream import rule_items, text_items, tokens
from firac_appeals.utils.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simple non-LLM baselines for downstream tasks.")
    parser.add_argument(
        "--task",
        required=True,
        choices=["outcome_prediction", "issue_generation", "rule_recommendation"],
    )
    parser.add_argument("--train", default="", help="Train task JSONL")
    parser.add_argument("--eval", required=True, help="Eval task JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=5, help="Number of rules to recommend")
    parser.add_argument("--max-candidates", type=int, default=3000, help="Rule candidates to score per query")
    args = parser.parse_args()

    train = read_jsonl(args.train) if args.train else []
    eval_rows = read_jsonl(args.eval)
    if args.task == "outcome_prediction":
        predictions = outcome_majority(train or eval_rows, eval_rows)
    elif args.task == "issue_generation":
        predictions = issue_nearest_neighbor(train or eval_rows, eval_rows)
    else:
        predictions = rule_tfidf_retrieval(
            train or eval_rows,
            eval_rows,
            top_k=args.top_k,
            max_candidates=args.max_candidates,
        )
    write_jsonl(args.output, predictions)
    print(f"Wrote {len(predictions)} {args.task} baseline predictions to {args.output}")


def outcome_majority(train: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(str((row.get("target") or {}).get("label", "")) for row in train)
    label = counts.most_common(1)[0][0] if counts else "dismissed"
    return [
        {
            "case_id": row.get("case_id", ""),
            "task": "outcome_prediction",
            "baseline": "majority_label",
            "label": label,
        }
        for row in eval_rows
    ]


def issue_nearest_neighbor(train: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    train_index = [(row, set(tokens(input_text(row)))) for row in train]
    predictions = []
    for row in eval_rows:
        query = set(tokens(input_text(row)))
        best_row = max(train_index, key=lambda item: jaccard(query, item[1]))[0] if train_index else {}
        predictions.append(
            {
                "case_id": row.get("case_id", ""),
                "task": "issue_generation",
                "baseline": "nearest_neighbor_facts_jaccard",
                "issues": text_items((best_row.get("target") or {}).get("issues")),
            }
        )
    return predictions


def rule_tfidf_retrieval(
    train: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    top_k: int,
    max_candidates: int,
) -> list[dict[str, Any]]:
    candidates = collect_rule_candidates(train)
    doc_freq = defaultdict(int)
    inverted: dict[str, set[int]] = defaultdict(set)
    candidate_token_counts = []
    for idx, candidate in enumerate(candidates):
        counts = Counter(tokens(candidate["text"]))
        candidate_token_counts.append(counts)
        for token in set(counts):
            doc_freq[token] += 1
            inverted[token].add(idx)
    num_docs = max(len(candidates), 1)
    predictions = []
    for row in eval_rows:
        query_tokens = Counter(tokens(input_text(row)))
        candidate_ids = candidate_ids_for_query(query_tokens, doc_freq, inverted, num_docs, max_candidates)
        scored = []
        for candidate_id in candidate_ids:
            candidate = candidates[candidate_id]
            score = bm25_lite(query_tokens, candidate_token_counts[candidate_id], doc_freq, num_docs)
            if score > 0:
                scored.append((score, candidate))
        scored.sort(key=lambda item: item[0], reverse=True)
        rules = [candidate["rule"] for _, candidate in scored[:top_k]]
        predictions.append(
            {
                "case_id": row.get("case_id", ""),
                "task": "rule_recommendation",
                "baseline": "train_rule_tfidf_retrieval",
                "rules": rules,
            }
        )
    return predictions


def candidate_ids_for_query(
    query: Counter[str],
    doc_freq: dict[str, int],
    inverted: dict[str, set[int]],
    num_docs: int,
    max_candidates: int,
) -> list[int]:
    token_weights = []
    for token, count in query.items():
        df = doc_freq.get(token, 0)
        if not df:
            continue
        idf = math.log((num_docs + 1) / (df + 1)) + 1.0
        token_weights.append((idf * min(count, 3), token))
    token_weights.sort(reverse=True)
    candidate_scores: Counter[int] = Counter()
    for weight, token in token_weights[:80]:
        for candidate_id in inverted.get(token, set()):
            candidate_scores[candidate_id] += weight
    return [candidate_id for candidate_id, _ in candidate_scores.most_common(max_candidates)]


def collect_rule_candidates(train: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_ref: dict[str, dict[str, Any]] = {}
    for row in train:
        text = input_text(row)
        for rule in rule_items((row.get("target") or {}).get("rules")):
            key = " ".join(tokens(" ".join(str(rule.get(k, "")) for k in ["reference", "provision", "citation"])))
            if not key or key in by_ref:
                continue
            rule_text = " ".join(
                str(rule.get(k, "")) for k in ["reference", "provision", "case_name", "citation", "relevance"]
            )
            by_ref[key] = {"rule": rule, "text": f"{rule_text}\n{text}"}
    return list(by_ref.values())


def bm25_lite(
    query: Counter[str],
    document: Counter[str],
    doc_freq: dict[str, int],
    num_docs: int,
) -> float:
    score = 0.0
    for token, q_count in query.items():
        tf = document.get(token, 0)
        if not tf:
            continue
        idf = math.log((num_docs + 1) / (doc_freq.get(token, 0) + 1)) + 1.0
        score += min(q_count, 3) * idf * (tf / (tf + 1.2))
    return score


def input_text(row: dict[str, Any]) -> str:
    input_data = row.get("input") or {}
    return str(input_data.get("input_text", ""))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


if __name__ == "__main__":
    main()
