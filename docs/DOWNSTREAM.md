# Downstream Experiments

## Common task files

Tasks are derived directly from released FIRAC records:

```bash
uv run firac-build-tasks \
  --input annotations.jsonl --output-dir outputs/tasks --split test
```

For LJP, construct the conditional binary task before inference:

```bash
uv run python scripts/build_binary_outcome_tasks.py \
  --input outputs/tasks/outcome_prediction.jsonl \
  --output outputs/tasks/outcome_prediction_binary.jsonl
```

Only `allowed` and `dismissed` are retained; heterogeneous `other` outcomes are
excluded. All systems are evaluated on the same filtered rows using Facts only.

## Issue identification

Input is Facts; targets are FIRAC Issues. Reported rows include Jaccard nearest
neighbor, MPNet dense nearest neighbor, and four zero-shot LLMs. All predictions
were rescored with `sentence-transformers/paraphrase-MiniLM-L6-v2` to avoid
evaluating MPNet retrieval with its own embedding space. Metrics are semantic
precision/recall/F1 and average-best token F1.

Relevant commands:

```bash
uv run firac-run-baseline --help
uv run python scripts/run_dense_retrieval_baselines.py --help
uv run python scripts/rescore_downstream_llm_experiment.py --help
```

Exact predictions and final rescored metrics are in
`artifacts/downstream/issue/`.

## Authority recommendation

Input is Facts plus Issues; targets are FIRAC Rules/Authorities. Baselines are a
lexical BM25-like retriever and MPNet dense retrieval. Zero-shot rows use Qwen,
Mistral, and Saul-family models. Mistral ablations replace gold Issues with
predicted Issues and optionally restrict generation to retrieved candidates.

Strict P@1, R@5, and nDCG@5 require exact equality after lowercasing and
alphanumeric normalization of the concatenated identifier fields. They are
conservative lower bounds, not canonical legal-entity matching. Token F1 exposes
partial lexical agreement. Commands and reported outputs are under
`scripts/run_dense_retrieval_baselines.py`,
`scripts/build_authority_ablation_tasks.py`, and
`artifacts/downstream/authority/`.

Zero-shot experiments can be rerun for one or more tasks with
`uv run firac-run-llm-eval --help`; exact model settings are indexed under
`configs/evaluation/`.

## Binary legal judgment prediction

Reported systems include majority prediction, zero-shot Qwen/Mistral/Saul,
word+character TF-IDF logistic regression, and `law-ai/InLegalBERT`.

TF-IDF+LR uses word 1-2 grams and `char_wb` 3-5 grams (up to 150,000 each),
sublinear TF-IDF, and class-balanced logistic regression (`C=1`, `liblinear`).

InLegalBERT is trained separately for each source with three seeds
(20260604-20260606), a stratified 90/10 train/validation split, 512-token
truncation, batch size 32, three epochs, AdamW at 2e-5, weight decay 0.01, 10%
linear warmup, class-weighted cross entropy, gradient clipping at 1.0, and BF16.
The best validation Macro-F1 checkpoint is evaluated once on the test set.

```bash
uv run python scripts/run_tfidf_outcome_classifier.py --help
uv run python scripts/run_outcome_classifier.py --help
uv run python scripts/audit_outcome_signal.py --help
```

Predictions, per-seed metrics, lexical coefficients, overlap audits, and paired
bootstrap comparisons are in `artifacts/downstream/ljp/`. Checkpoints are
excluded because they are large and reproducible from the public base model.
Paired confidence intervals and system comparisons can be regenerated with
`scripts/bootstrap_outcome_comparison.py`.

## Summarization

The released predictions can be rescored with
`uv run firac-score-summaries --help`; the metric specification is
`configs/evaluation/summarization.yaml` and exact outputs are under
`artifacts/summarization/`.
