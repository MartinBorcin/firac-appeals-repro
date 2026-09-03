from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from statistics import mean
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
RULE_RANK_CUTOFFS = (1, 3, 5, 10, 20, 50, 100)


def build_task_examples(rows: list[dict[str, Any]], task: str) -> list[dict[str, Any]]:
    builders = {
        "outcome_prediction": outcome_prediction_example,
        "issue_generation": issue_generation_example,
        "rule_recommendation": rule_recommendation_example,
    }
    if task not in builders:
        raise ValueError(f"Unknown downstream task: {task}")
    examples = []
    for row in rows:
        example = builders[task](row)
        if example is not None:
            examples.append(example)
    return examples


def outcome_prediction_example(row: dict[str, Any]) -> dict[str, Any] | None:
    label = normalized_outcome(row)
    if label not in {"allowed", "dismissed", "other"}:
        return None
    facts = text_items((row.get("firac") or {}).get("facts"))
    if not facts:
        return None
    return base_example(row, "outcome_prediction") | {
        "input": {
            "facts": facts,
            "input_text": "\n".join(f"- {fact}" for fact in facts),
        },
        "target": {"label": label},
    }


def issue_generation_example(row: dict[str, Any]) -> dict[str, Any] | None:
    firac = row.get("firac") or {}
    facts = text_items(firac.get("facts"))
    issues = text_items(firac.get("issues"))
    if not facts or not issues:
        return None
    return base_example(row, "issue_generation") | {
        "input": {
            "facts": facts,
            "input_text": "\n".join(f"- {fact}" for fact in facts),
        },
        "target": {"issues": issues},
    }


def rule_recommendation_example(row: dict[str, Any]) -> dict[str, Any] | None:
    firac = row.get("firac") or {}
    facts = text_items(firac.get("facts"))
    issues = text_items(firac.get("issues"))
    rules = rule_items(firac.get("rules"))
    if not facts or not rules:
        return None
    issue_text = "\n".join(f"- {issue}" for issue in issues)
    facts_text = "\n".join(f"- {fact}" for fact in facts)
    input_text = f"Facts:\n{facts_text}"
    if issue_text:
        input_text += f"\n\nIssues:\n{issue_text}"
    return base_example(row, "rule_recommendation") | {
        "input": {
            "facts": facts,
            "issues": issues,
            "input_text": input_text,
        },
        "target": {"rules": rules},
    }


def base_example(row: dict[str, Any], task: str) -> dict[str, Any]:
    return {
        "case_id": row.get("case_id", ""),
        "task": task,
        "source_dataset": row.get("source_dataset", ""),
        "split": row.get("split", ""),
        "metadata": {
            "outcome_label": row.get("outcome_label", ""),
            "judgment_chars": len(str(row.get("judgment_text", ""))),
            "facts_count": len(text_items((row.get("firac") or {}).get("facts"))),
            "issues_count": len(text_items((row.get("firac") or {}).get("issues"))),
            "rules_count": len(rule_items((row.get("firac") or {}).get("rules"))),
        },
    }


def evaluate_predictions(
    task: str,
    references: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    semantic_model_name: str | None = None,
    semantic_device: str | None = None,
    semantic_batch_size: int = 32,
) -> dict[str, Any]:
    pred_by_id = {str(row.get("case_id", "")): row for row in predictions}
    per_case = []
    issue_semantic_scorer = None
    if task == "issue_generation" and semantic_model_name:
        issue_semantic_scorer = IssueSemanticScorer(
            semantic_model_name,
            device=semantic_device,
            batch_size=semantic_batch_size,
        )
        issue_semantic_scorer.preload(issue_texts_for_semantic_scoring(references, predictions))
    for ref in references:
        case_id = str(ref.get("case_id", ""))
        pred = pred_by_id.get(case_id, {})
        if task == "outcome_prediction":
            scores = evaluate_outcome(ref, pred)
        elif task == "issue_generation":
            scores = evaluate_issue_generation(ref, pred, semantic_scorer=issue_semantic_scorer)
        elif task == "rule_recommendation":
            scores = evaluate_rule_recommendation(ref, pred)
        else:
            raise ValueError(f"Unknown downstream task: {task}")
        per_case.append({"case_id": case_id, **scores})
    return {
        "task": task,
        "count": len(per_case),
        "aggregate": aggregate(per_case),
        "per_case": per_case,
    }


def issue_texts_for_semantic_scoring(
    references: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> list[str]:
    pred_by_id = {str(row.get("case_id", "")): row for row in predictions}
    texts: set[str] = set()
    for ref in references:
        pred = pred_by_id.get(str(ref.get("case_id", "")), {})
        for item in text_items(ref.get("target", {}).get("issues")):
            text = normalize_display_text(item)
            if text:
                texts.add(text)
        for item in predicted_text_items(pred, "issues"):
            text = normalize_display_text(item)
            if text:
                texts.add(text)
    return sorted(texts)


def evaluate_outcome(reference: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    gold = str(reference.get("target", {}).get("label", "")).lower()
    pred = extract_predicted_label(prediction)
    return {
        "gold": gold,
        "predicted": pred,
        "accuracy": 1.0 if pred == gold else 0.0,
    }


def evaluate_issue_generation(
    reference: dict[str, Any],
    prediction: dict[str, Any],
    semantic_scorer: "IssueSemanticScorer | None" = None,
) -> dict[str, Any]:
    gold = text_items(reference.get("target", {}).get("issues"))
    pred = predicted_text_items(prediction, "issues")
    scores = set_generation_scores(gold, pred, prefix="issue")
    if semantic_scorer is not None:
        scores.update(semantic_scorer.score(gold, pred, prefix="issue_semantic"))
    return scores


def evaluate_rule_recommendation(reference: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    gold_rules = rule_items(reference.get("target", {}).get("rules"))
    pred_rules = predicted_rule_items(prediction)
    gold_refs = [normalize_rule_ref(rule) for rule in gold_rules if normalize_rule_ref(rule)]
    pred_refs = [normalize_rule_ref(rule) for rule in pred_rules if normalize_rule_ref(rule)]
    gold_set = set(gold_refs)
    pred_set = set(pred_refs)
    true_positive = len(gold_set & pred_set)
    precision = true_positive / max(len(pred_set), 1)
    recall = true_positive / max(len(gold_set), 1)
    f1 = f_score(precision, recall)
    out = {
        "rule_reference_precision": precision,
        "rule_reference_recall": recall,
        "rule_reference_f1": f1,
        "rule_reference_exact_matches": true_positive,
        "gold_rule_count": len(gold_set),
        "predicted_rule_count": len(pred_set),
    }
    for k in RULE_RANK_CUTOFFS:
        out[f"rule_recall_at_{k}"] = recall_at_k(gold_refs, pred_refs, k)
        out[f"rule_precision_at_{k}"] = precision_at_k(gold_refs, pred_refs, k)
        out[f"rule_f1_at_{k}"] = f_score(out[f"rule_precision_at_{k}"], out[f"rule_recall_at_{k}"])
        out[f"rule_hit_rate_at_{k}"] = hit_rate_at_k(gold_refs, pred_refs, k)
        out[f"rule_average_precision_at_{k}"] = average_precision_at_k(gold_refs, pred_refs, k)
        out[f"rule_reciprocal_rank_at_{k}"] = reciprocal_rank_at_k(gold_refs, pred_refs, k)
        out[f"rule_ndcg_at_{k}"] = ndcg_at_k(gold_refs, pred_refs, k)
    out["rule_average_precision"] = average_precision_at_k(gold_refs, pred_refs, len(pred_refs))
    out["rule_reciprocal_rank"] = reciprocal_rank_at_k(gold_refs, pred_refs, len(pred_refs))
    out["rule_r_precision"] = precision_at_k(gold_refs, pred_refs, len(gold_set))
    out.update(set_generation_scores(gold_refs, pred_refs, prefix="rule_text"))
    return out


def set_generation_scores(gold: list[str], pred: list[str], prefix: str) -> dict[str, float]:
    gold_norm = [normalize_text(item) for item in gold if normalize_text(item)]
    pred_norm = [normalize_text(item) for item in pred if normalize_text(item)]
    gold_set = set(gold_norm)
    pred_set = set(pred_norm)
    exact = len(gold_set & pred_set)
    precision = exact / max(len(pred_set), 1)
    recall = exact / max(len(gold_set), 1)
    token_scores = [max_token_f1(g, pred_norm) for g in gold_norm] if gold_norm else [0.0]
    return {
        f"{prefix}_exact_precision": precision,
        f"{prefix}_exact_recall": recall,
        f"{prefix}_exact_f1": f_score(precision, recall),
        f"{prefix}_avg_best_token_f1": mean(token_scores),
        f"gold_{prefix}_count": len(gold_norm),
        f"predicted_{prefix}_count": len(pred_norm),
    }


def aggregate(per_case: list[dict[str, Any]]) -> dict[str, Any]:
    numeric: dict[str, list[float]] = defaultdict(list)
    labels_gold = []
    labels_pred = []
    for row in per_case:
        for key, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric[key].append(float(value))
        if "gold" in row and "predicted" in row:
            labels_gold.append(str(row["gold"]))
            labels_pred.append(str(row["predicted"]))
    out = {key: mean(values) for key, values in numeric.items() if values}
    if labels_gold:
        out["macro_f1"] = macro_f1(labels_gold, labels_pred)
        out["label_distribution"] = dict(Counter(labels_gold))
        out["prediction_distribution"] = dict(Counter(labels_pred))
    return out


def macro_f1(gold: list[str], pred: list[str]) -> float:
    labels = sorted(set(gold) | set(pred))
    scores = []
    for label in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        scores.append(f_score(precision, recall))
    return mean(scores) if scores else 0.0


def text_items(value: Any) -> list[str]:
    items = []
    for item in as_list(value):
        if isinstance(item, dict):
            text = first_text(item, ["question", "issue", "text", "reference", "label"])
        else:
            text = str(item or "")
        text = normalize_display_text(text)
        if text:
            items.append(text)
    return items


def rule_items(value: Any) -> list[dict[str, Any]]:
    items = []
    for item in as_list(value):
        if isinstance(item, dict):
            reference = normalize_display_text(item.get("reference") or item.get("case_name") or item.get("citation"))
            if not reference:
                continue
            items.append(
                {
                    "type": str(item.get("type", "other") or "other"),
                    "reference": reference,
                    "provision": normalize_display_text(item.get("provision", "")),
                    "case_name": normalize_display_text(item.get("case_name", "")),
                    "citation": normalize_display_text(item.get("citation", "")),
                    "treatment_type": normalize_display_text(item.get("treatment_type", "")),
                    "relevance": normalize_display_text(item.get("relevance", "")),
                }
            )
        else:
            reference = normalize_display_text(item)
            if reference:
                items.append({"type": "other", "reference": reference})
    return items


def predicted_text_items(prediction: dict[str, Any], key: str) -> list[str]:
    if key in prediction:
        return text_items(prediction[key])
    if "prediction" in prediction:
        try:
            parsed = json.loads(str(prediction["prediction"]))
            return text_items(parsed.get(key, parsed))
        except Exception:
            return split_prediction_lines(str(prediction["prediction"]))
    return []


def predicted_rule_items(prediction: dict[str, Any]) -> list[dict[str, Any]]:
    if "rules" in prediction:
        return rule_items(prediction["rules"])
    if "prediction" in prediction:
        try:
            parsed = json.loads(str(prediction["prediction"]))
            return rule_items(parsed.get("rules", parsed))
        except Exception:
            return [{"type": "other", "reference": item} for item in split_prediction_lines(str(prediction["prediction"]))]
    return []


def extract_predicted_label(prediction: dict[str, Any]) -> str:
    for key in ["label", "predicted", "prediction"]:
        if key in prediction:
            text = str(prediction[key]).lower()
            for label in ["allowed", "dismissed", "other"]:
                if label in text:
                    return label
    return ""


def normalized_outcome(row: dict[str, Any]) -> str:
    firac = row.get("firac") or {}
    conclusion = firac.get("conclusion") if isinstance(firac, dict) else {}
    if isinstance(conclusion, dict):
        simple = str(conclusion.get("simplified_label", "")).lower().strip()
        if simple in {"allowed", "dismissed", "other"}:
            return simple
    label = str(row.get("outcome_label", "")).lower().strip()
    if label in {"allowed", "dismissed", "other"}:
        return label
    return "other" if label else ""


def normalize_rule_ref(rule: dict[str, Any]) -> str:
    parts = [
        rule.get("reference", ""),
        rule.get("provision", ""),
        rule.get("case_name", ""),
        rule.get("citation", ""),
    ]
    return normalize_text(" ".join(str(part) for part in parts if part))


def recall_at_k(gold: list[str], pred: list[str], k: int) -> float:
    gold_set = set(gold)
    if k <= 0 or not gold_set:
        return 0.0
    return len(gold_set & set(pred[:k])) / len(gold_set)


def precision_at_k(gold: list[str], pred: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(gold) & set(pred[:k])) / k


def hit_rate_at_k(gold: list[str], pred: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return 1.0 if set(gold) & set(pred[:k]) else 0.0


def average_precision_at_k(gold: list[str], pred: list[str], k: int) -> float:
    gold_set = set(gold)
    if k <= 0 or not gold_set:
        return 0.0
    seen: set[str] = set()
    score = 0.0
    hits = 0
    for idx, item in enumerate(pred[:k], start=1):
        if item in gold_set and item not in seen:
            hits += 1
            score += hits / idx
            seen.add(item)
    return score / min(len(gold_set), k)


def reciprocal_rank_at_k(gold: list[str], pred: list[str], k: int) -> float:
    gold_set = set(gold)
    if k <= 0 or not gold_set:
        return 0.0
    for idx, item in enumerate(pred[:k], start=1):
        if item in gold_set:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(gold: list[str], pred: list[str], k: int) -> float:
    gold_set = set(gold)
    if k <= 0 or not gold_set:
        return 0.0
    dcg = 0.0
    seen: set[str] = set()
    for idx, item in enumerate(pred[:k], start=1):
        if item in gold_set and item not in seen:
            dcg += 1.0 / math.log2(idx + 1)
            seen.add(item)
    ideal_hits = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


class IssueSemanticScorer:
    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 32) -> None:
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "Semantic issue metrics require the eval extra: "
                "uv sync --extra eval or uv add --optional eval sentence-transformers"
            ) from exc

        self.np = np
        kwargs = {"device": device} if device else {}
        self.model = SentenceTransformer(model_name, **kwargs)
        self.batch_size = batch_size
        self.cache: dict[str, Any] = {}

    def score(self, gold: list[str], pred: list[str], prefix: str) -> dict[str, float]:
        gold_clean = [normalize_display_text(item) for item in gold if normalize_display_text(item)]
        pred_clean = [normalize_display_text(item) for item in pred if normalize_display_text(item)]
        if not gold_clean or not pred_clean:
            return {
                f"{prefix}_precision": 0.0,
                f"{prefix}_recall": 0.0,
                f"{prefix}_f1": 0.0,
                f"{prefix}_avg_best_cosine": 0.0,
            }
        gold_emb = self.encode(gold_clean)
        pred_emb = self.encode(pred_clean)
        sim = gold_emb @ pred_emb.T
        recall = float(self.np.mean(self.np.max(sim, axis=1)))
        precision = float(self.np.mean(self.np.max(sim, axis=0)))
        return {
            f"{prefix}_precision": finite_float(precision),
            f"{prefix}_recall": finite_float(recall),
            f"{prefix}_f1": f_score(precision, recall),
            f"{prefix}_avg_best_cosine": finite_float((precision + recall) / 2),
        }

    def encode(self, texts: list[str]) -> Any:
        missing = [text for text in texts if text not in self.cache]
        if missing:
            embeddings = self.model.encode(
                missing,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for text, embedding in zip(missing, embeddings):
                self.cache[text] = self.np.asarray(embedding, dtype=self.np.float32)
        return self.np.vstack([self.cache[text] for text in texts])

    def preload(self, texts: list[str]) -> None:
        unique_texts = [text for text in dict.fromkeys(texts) if text and text not in self.cache]
        if unique_texts:
            self.encode(unique_texts)


def max_token_f1(gold: str, predictions: list[str]) -> float:
    if not predictions:
        return 0.0
    return max(token_f1(gold, pred) for pred in predictions)


def token_f1(a: str, b: str) -> float:
    a_counts = Counter(tokens(a))
    b_counts = Counter(tokens(b))
    overlap = sum((a_counts & b_counts).values())
    precision = overlap / max(sum(b_counts.values()), 1)
    recall = overlap / max(sum(a_counts.values()), 1)
    return f_score(precision, recall)


def f_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def split_prediction_lines(text: str) -> list[str]:
    items = []
    for line in text.splitlines():
        line = re.sub(r"^\s*[-*\d.)]+\s*", "", line).strip()
        if line:
            items.append(line)
    if items:
        return items
    text = text.strip()
    return [text] if text else []


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def first_text(value: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        if value.get(key):
            return str(value[key])
    return ""


def normalize_display_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_text(text: Any) -> str:
    return " ".join(tokens(str(text or "")))


def tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


def finite_float(value: float) -> float:
    return value if math.isfinite(value) else 0.0
