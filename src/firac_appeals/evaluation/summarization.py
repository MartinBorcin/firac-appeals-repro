from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from firac_appeals.utils.io import ensure_parent, write_jsonl


TOKEN_RE = re.compile(r"\w+")
SPACE_RE = re.compile(r"\s+")


def evaluate_rows(
    rows: list[dict[str, Any]],
    *,
    summary_variant: str = "firac_inabs_headnote",
    compute_bertscore: bool = False,
    bertscore_batch_size: int = 4,
    bertscore_model_type: str | None = None,
) -> dict[str, Any]:
    per_case: list[dict[str, Any]] = []
    predictions: list[str] = []
    references: list[str] = []
    for row in rows:
        source = row.get("judgment_text", "")
        reference = row.get("reference_summary", "")
        prediction = row.get("prediction") or row.get("firac_summary") or render_summary(row, summary_variant)
        metrics = summarization_metrics(source, prediction, reference)
        metrics["case_id"] = row.get("case_id", "")
        metrics["split"] = row.get("split", "")
        per_case.append(metrics)
        predictions.append(prediction)
        references.append(reference)

    if compute_bertscore and predictions:
        add_bertscore(
            per_case,
            predictions,
            references,
            batch_size=bertscore_batch_size,
            model_type=bertscore_model_type,
        )

    aggregate = aggregate_metrics(per_case)
    return {
        "summary_variant": summary_variant,
        "aggregate": aggregate,
        "per_case": per_case,
        "count": len(per_case),
    }


def prediction_rows(rows: list[dict[str, Any]], *, summary_variant: str = "firac_inabs_headnote") -> list[dict[str, Any]]:
    rendered = []
    for row in rows:
        prediction = row.get("prediction") or row.get("firac_summary") or render_summary(row, summary_variant)
        rendered.append(
            {
                "case_id": row.get("case_id", ""),
                "split": row.get("split", ""),
                "prediction": prediction,
                "reference_summary": row.get("reference_summary", ""),
                "prediction_words": len(tokens(prediction)),
                "reference_words": len(tokens(row.get("reference_summary", ""))),
            }
        )
    return rendered


def render_summary(row: dict[str, Any], summary_variant: str = "firac_inabs_headnote") -> str:
    if summary_variant == "firac_inabs_headnote":
        return firac_headnote_summary(row)
    if summary_variant == "firac_inabs_headnote_l_capped":
        return firac_headnote_summary_l_capped(row)
    raise ValueError(f"Unknown summary variant: {summary_variant}")


def firac_headnote_summary(row: dict[str, Any]) -> str:
    """Render FIRAC as an IN-Abs-style legal headnote.

    IN-Abs references generally start with the material facts/procedural posture
    and then move to a "Held" statement that combines the result, legal rules,
    and application. This renderer follows that shape without asking another
    model to summarize the annotation.
    """
    firac = row.get("firac") or {}
    if not isinstance(firac, dict):
        return normalize_text(firac)

    facts = render_facts(firac.get("facts"))
    issues = render_issues(firac.get("issues"))
    conclusion = render_conclusion(firac.get("conclusion"))
    rules = render_rules(firac.get("rules"))
    analysis = render_analysis(firac.get("analysis"))

    paragraphs = []
    if facts:
        paragraphs.append(facts)
    if issues:
        paragraphs.append(f"The question before the Court was {issues}.")

    held_parts = [part for part in [conclusion, rules, analysis] if part]
    if held_parts:
        paragraphs.append("Held: " + " ".join(held_parts))

    return normalize_text("\n\n".join(paragraphs))


def firac_headnote_summary_l_capped(row: dict[str, Any]) -> str:
    full = firac_headnote_summary(row)
    reference_len = len(tokens(row.get("reference_summary", "")))
    if reference_len <= 0 or len(tokens(full)) <= reference_len:
        return full

    sections = firac_headnote_sections(row)
    budget = reference_len
    used: list[SummaryUnit] = []

    weights = {
        "facts": 0.40,
        "issue": 0.08,
        "conclusion": 0.12,
        "rules": 0.20,
        "analysis": 0.20,
    }
    # Keep the shortest high-value bridge/result fields before spending the
    # larger facts/rules/application budgets.
    primary_order = ["facts", "issue", "conclusion", "rules", "analysis"]
    deferred: list[SummaryUnit] = []
    remaining = budget

    for section in primary_order:
        section_units = sections.get(section, [])
        if not section_units:
            continue
        soft_budget = max(1, math.floor(budget * weights[section]))
        spent = 0
        for unit in section_units:
            if unit.words <= max(0, soft_budget - spent) and unit.words <= remaining:
                used.append(unit)
                spent += unit.words
                remaining -= unit.words
            else:
                deferred.append(unit)

    for unit in original_unit_order(sections):
        if remaining <= 0:
            break
        if unit in used or unit not in deferred:
            continue
        if unit.words <= remaining:
            used.append(unit)
            remaining -= unit.words

    if remaining > 0:
        for unit in original_unit_order(sections):
            if unit in used:
                continue
            trimmed = trim_words(unit.text, remaining)
            if trimmed:
                used.append(SummaryUnit(unit.section, trimmed))
            break

    return assemble_headnote_units(used, budget)


class SummaryUnit:
    def __init__(self, section: str, text: str) -> None:
        self.section = section
        self.text = normalize_text(text)
        self.words = len(tokens(self.text))


def firac_headnote_sections(row: dict[str, Any]) -> dict[str, list[SummaryUnit]]:
    firac = row.get("firac") or {}
    if not isinstance(firac, dict):
        return {"facts": [SummaryUnit("facts", normalize_text(firac))]}
    sections: dict[str, list[SummaryUnit]] = {
        "facts": [SummaryUnit("facts", ensure_sentence(item)) for item in text_items(firac.get("facts"))],
        "issue": [],
        "conclusion": [],
        "rules": [],
        "analysis": [],
    }
    issues = render_issues(firac.get("issues"))
    if issues:
        sections["issue"].append(SummaryUnit("issue", f"The question before the Court was {issues}."))
    conclusion = render_conclusion(firac.get("conclusion"))
    if conclusion:
        sections["conclusion"].append(SummaryUnit("conclusion", conclusion))
    for rule in render_rule_units(firac.get("rules")):
        sections["rules"].append(SummaryUnit("rules", rule))
    for analysis in render_analysis_units(firac.get("analysis")):
        sections["analysis"].append(SummaryUnit("analysis", analysis))
    return sections


def original_unit_order(sections: dict[str, list[SummaryUnit]]) -> list[SummaryUnit]:
    units: list[SummaryUnit] = []
    for section in ["facts", "issue", "conclusion", "rules", "analysis"]:
        units.extend(sections.get(section, []))
    return units


def assemble_headnote_units(units: list[SummaryUnit], word_budget: int) -> str:
    facts = [unit.text for unit in units if unit.section == "facts"]
    issue = [unit.text for unit in units if unit.section == "issue"]
    conclusion = [unit.text for unit in units if unit.section == "conclusion"]
    rules = [unit.text for unit in units if unit.section == "rules"]
    analysis = [unit.text for unit in units if unit.section == "analysis"]

    paragraphs = []
    if facts:
        paragraphs.append(" ".join(facts))
    if issue:
        paragraphs.append(" ".join(issue))
    held_parts = []
    if conclusion:
        held_parts.append(" ".join(conclusion))
    if rules:
        held_parts.append("The Court considered " + " ".join(rules))
    if analysis:
        held_parts.append("Applying those rules, " + " ".join(analysis))
    if held_parts:
        paragraphs.append("Held: " + " ".join(held_parts))
    return trim_words(normalize_text("\n\n".join(paragraphs)), word_budget)


def render_facts(value: Any) -> str:
    items = text_items(value)
    if not items:
        return ""
    return normalize_text(" ".join(ensure_sentence(item) for item in items))


def render_issues(value: Any) -> str:
    items = []
    for item in as_list(value):
        if isinstance(item, dict):
            items.append(first_text(item, ["question", "issue", "text"]))
        else:
            items.append(flatten_text(item))
    items = [strip_terminal_punct(item) for item in items if normalize_text(item)]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return "; and ".join(items)


def render_conclusion(value: Any) -> str:
    if isinstance(value, dict):
        text = first_text(value, ["text", "label", "outcome"])
        label = normalize_text(value.get("label", ""))
        if label and label.lower() not in text.lower():
            return ensure_sentence(f"{label}: {text}")
        return ensure_sentence(text)
    return ensure_sentence(flatten_text(value))


def render_rules(value: Any) -> str:
    rendered = render_rule_units(value)
    if not rendered:
        return ""
    return "The Court considered " + " ".join(rendered)


def render_rule_units(value: Any) -> list[str]:
    rendered = []
    for item in as_list(value):
        if isinstance(item, dict):
            ref = first_text(item, ["reference", "provision", "citation", "case_name", "type"])
            treatment = first_text(item, ["treatment_type", "treatment"])
            relevance = first_text(item, ["relevance", "explanation", "principle", "holding"])
            if ref and relevance:
                prefix = ref
                if treatment and treatment.lower() not in {"not applicable", "n/a", "none"}:
                    prefix = f"{prefix} ({treatment})"
                rendered.append(ensure_sentence(f"{prefix}: {relevance}"))
            else:
                rendered.append(ensure_sentence(flatten_text(item)))
        else:
            rendered.append(ensure_sentence(flatten_text(item)))
    return [item for item in rendered if normalize_text(item)]


def render_analysis(value: Any) -> str:
    parts = render_analysis_units(value)
    if not parts:
        return ""
    return "Applying those rules, " + " ".join(parts)


def render_analysis_units(value: Any) -> list[str]:
    parts = []
    for item in as_list(value):
        if isinstance(item, dict):
            steps = [ensure_sentence(step) for step in text_items(item.get("steps"))]
            conclusion = ensure_sentence(first_text(item, ["conclusion", "holding", "result"]))
            body = " ".join(step for step in steps if step)
            if conclusion:
                body = f"{body} {conclusion}".strip()
            if body:
                issue = strip_terminal_punct(first_text(item, ["issue", "question"]))
                if issue:
                    parts.append(f"On {issue}, {lower_first(body)}")
                else:
                    parts.append(body)
        else:
            text = ensure_sentence(flatten_text(item))
            if text:
                parts.append(text)
    return parts


def summarization_metrics(source: str, prediction: str, reference: str = "") -> dict[str, float]:
    source_toks = tokens(source)
    pred_toks = tokens(prediction)
    ref_toks = tokens(reference)
    metrics = {
        "source_words": float(len(source_toks)),
        "prediction_words": float(len(pred_toks)),
        "reference_words": float(len(ref_toks)),
        "compression_ratio": safe_div(len(pred_toks), len(source_toks)),
        "reference_length_ratio": safe_div(len(pred_toks), len(ref_toks)),
        "novel_unigrams": novel_ngram_rate(source_toks, pred_toks, 1),
        "novel_bigrams": novel_ngram_rate(source_toks, pred_toks, 2),
        "novel_trigrams": novel_ngram_rate(source_toks, pred_toks, 3),
    }
    coverage, density = extractive_coverage_density(source_toks, pred_toks)
    metrics["extractive_coverage"] = coverage
    metrics["extractive_density"] = density
    metrics.update(rouge_metrics(prediction, reference))
    return metrics


def rouge_metrics(prediction: str, reference: str) -> dict[str, float]:
    if not normalize_text(reference):
        return {
            "rouge1_precision": math.nan,
            "rouge1_recall": math.nan,
            "rouge1_f1": math.nan,
            "rouge2_precision": math.nan,
            "rouge2_recall": math.nan,
            "rouge2_f1": math.nan,
            "rougeL_precision": math.nan,
            "rougeL_recall": math.nan,
            "rougeL_f1": math.nan,
        }
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    scores = scorer.score(reference, prediction)
    out = {}
    for name, score in scores.items():
        out[f"{name}_precision"] = score.precision
        out[f"{name}_recall"] = score.recall
        out[f"{name}_f1"] = score.fmeasure
    return out


def add_bertscore(
    per_case: list[dict[str, Any]],
    predictions: list[str],
    references: list[str],
    *,
    batch_size: int,
    model_type: str | None,
) -> None:
    patch_transformers_tokenizer_max_len()
    from bert_score import score

    kwargs: dict[str, Any] = {
        "cands": predictions,
        "refs": references,
        "lang": "en",
        "batch_size": batch_size,
        "verbose": True,
    }
    if model_type:
        kwargs["model_type"] = model_type
    precision, recall, f1 = score(**kwargs)
    for row, p, r, f in zip(per_case, precision.tolist(), recall.tolist(), f1.tolist(), strict=True):
        row["bertscore_precision"] = float(p)
        row["bertscore_recall"] = float(r)
        row["bertscore_f1"] = float(f)


def patch_transformers_tokenizer_max_len() -> None:
    """Keep bert-score 0.3.4 compatible with newer transformers.

    IN-Abs used bert-score 0.3.4. That release accesses tokenizer.max_len,
    while recent transformers exposes model_max_length instead.
    """
    try:
        from transformers import PreTrainedTokenizerBase
    except Exception:
        return
    if not hasattr(PreTrainedTokenizerBase, "max_len"):
        PreTrainedTokenizerBase.max_len = property(lambda self: self.model_max_length)  # type: ignore[attr-defined]


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def text_items(value: Any) -> list[str]:
    items = []
    for item in as_list(value):
        text = flatten_text(item)
        if normalize_text(text):
            items.append(normalize_text(text))
    return items


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(flatten_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(flatten_text(item) for item in value.values())
    return str(value)


def first_text(value: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        if key in value and normalize_text(value[key]):
            return normalize_text(value[key])
    return ""


def normalize_text(text: Any) -> str:
    return SPACE_RE.sub(" ", str(text or "")).strip()


def ensure_sentence(text: str) -> str:
    text = normalize_text(text)
    if not text:
        return ""
    if text[-1] in ".?!":
        return text
    return text + "."


def strip_terminal_punct(text: str) -> str:
    return normalize_text(text).rstrip(" .?!;:")


def lower_first(text: str) -> str:
    text = normalize_text(text)
    if not text:
        return ""
    return text[0].lower() + text[1:]


def trim_words(text: str, max_words: int) -> str:
    words = normalize_text(text).split()
    if max_words <= 0:
        return ""
    if len(tokens(text)) <= max_words:
        return normalize_text(text)
    trimmed_words = words[:max_words]
    trimmed = " ".join(trimmed_words).rstrip(" ,;:")
    while trimmed_words and len(tokens(trimmed)) > max_words:
        trimmed_words = trimmed_words[:-1]
        trimmed = " ".join(trimmed_words).rstrip(" ,;:")
    return ensure_sentence(trimmed)


def novel_ngram_rate(source_toks: list[str], pred_toks: list[str], n: int) -> float:
    pred = ngrams(pred_toks, n)
    if not pred:
        return 0.0
    source = set(ngrams(source_toks, n))
    return sum(1 for gram in pred if gram not in source) / len(pred)


def ngrams(items: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(items[i : i + n]) for i in range(0, max(len(items) - n + 1, 0))]


def extractive_coverage_density(source_toks: list[str], pred_toks: list[str]) -> tuple[float, float]:
    if not source_toks or not pred_toks:
        return 0.0, 0.0
    source_positions: dict[str, list[int]] = {}
    for i, tok in enumerate(source_toks):
        source_positions.setdefault(tok, []).append(i)

    fragments: list[int] = []
    i = 0
    while i < len(pred_toks):
        best = 0
        for j in source_positions.get(pred_toks[i], []):
            length = 0
            while (
                i + length < len(pred_toks)
                and j + length < len(source_toks)
                and pred_toks[i + length] == source_toks[j + length]
            ):
                length += 1
            best = max(best, length)
        if best:
            fragments.append(best)
            i += best
        else:
            i += 1
    coverage = safe_div(sum(fragments), len(pred_toks))
    density = safe_div(sum(length * length for length in fragments), len(pred_toks))
    return coverage, density


def aggregate_metrics(per_case: list[dict[str, Any]]) -> dict[str, float]:
    if not per_case:
        return {}
    keys = [k for k in per_case[0] if k not in {"case_id", "split"}]
    agg = {}
    for key in keys:
        values = [row[key] for row in per_case if isinstance(row.get(key), (int, float)) and not math.isnan(row[key])]
        agg[key] = mean(values) if values else math.nan
    return agg


def outcome_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        conclusion = ((row.get("firac") or {}).get("conclusion") or {}) if isinstance(row.get("firac"), dict) else {}
        if isinstance(conclusion, dict):
            counts[str(conclusion.get("simplified_label") or conclusion.get("label") or "unknown")] += 1
        else:
            counts["unknown"] += 1
    return dict(sorted(counts.items()))


def safe_div(num: float, denom: float) -> float:
    return 0.0 if denom == 0 else num / denom


def write_metrics(path: str | Path, result: dict[str, Any]) -> None:
    p = ensure_parent(path)
    p.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_predictions(path: str | Path, rows: list[dict[str, Any]], *, summary_variant: str = "firac_inabs_headnote") -> None:
    write_jsonl(path, prediction_rows(rows, summary_variant=summary_variant))
