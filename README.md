# FIRAC Appeals Reproducibility Package

This repository accompanies the FIRAC Appeals paper. It contains the minimal
code, prompts, configs, and tiny fixtures needed to run the FIRAC annotation
framework and reproduce evaluation workflows. It intentionally does not include
run artefacts, raw corpora, generated paper reports, orchestration scripts, logs, or local cluster utilities.

<img width="3269" height="861" alt="FIRAC_dataset_paper_figure-2" src="https://github.com/user-attachments/assets/eb0971ad-6275-44b1-a46f-53aeb0e75a72" />

## Installation

```bash
uv sync --extra eval
```

The project uses Python 3.12 and exposes all commands through `uv run`.

## Smoke Test

Run the tests and a credential-free mock annotation pass:

```bash
uv run pytest

uv run firac-annotate \
  --input data/fixtures/cases.jsonl \
  --output-dir outputs/mock_annotation \
  --config configs/annotation/mock.yaml
```

The mock backend returns deterministic JSON and is only for checking that the
pipeline, schemas, output paths, and prompts are wired correctly.

## Annotate Your Own Data

Normalize JSONL, JSON array, or CSV data into FIRAC case records:

```bash
uv run firac-prepare-cases \
  --input my_cases.jsonl \
  --output outputs/cases.jsonl \
  --text-field judgment_text \
  --id-field case_id
```

For supported paper source datasets, load common extracted/raw layouts directly:

```bash
uv run firac-load-source \
  --source in_abs \
  --input path/to/extracted_in_abs \
  --output outputs/in_abs_cases.jsonl

uv run firac-load-source \
  --source supreme_court_vanga \
  --input path/to/extracted_judgments \
  --metadata path/to/metadata.jsonl \
  --output outputs/supreme_court_cases.jsonl
```

Run annotation against any OpenAI-compatible chat-completions endpoint:

```bash
OPENAI_BASE_URL=https://api.example.com/v1 \
OPENAI_API_KEY=... \
FIRAC_MODEL=your-model-name \
uv run firac-annotate \
  --input outputs/cases.jsonl \
  --output-dir outputs/annotations \
  --config configs/annotation/openai_compatible.yaml
```

Annotation outputs are written to `final.jsonl`, with resumable per-case JSON
files under `cases/` when `write_case_files` is enabled.

## Reproduce Evaluation Workflows

Build downstream tasks from completed FIRAC annotations:

```bash
uv run firac-build-tasks \
  --input data/fixtures/annotations.jsonl \
  --output-dir outputs/tasks/test \
  --split test
```

Run and score a simple baseline:

```bash
uv run firac-run-baseline \
  --task outcome_prediction \
  --eval outputs/tasks/test/outcome_prediction.jsonl \
  --output outputs/baselines/outcome_prediction.jsonl

uv run firac-score-predictions \
  --task outcome_prediction \
  --references outputs/tasks/test/outcome_prediction.jsonl \
  --predictions outputs/baselines/outcome_prediction.jsonl \
  --output outputs/metrics/outcome_prediction.json
```

Run a zero-shot LLM evaluation:

```bash
OPENAI_BASE_URL=https://api.example.com/v1 \
OPENAI_API_KEY=... \
FIRAC_MODEL=your-model-name \
uv run firac-run-llm-eval \
  --task-dir outputs/tasks/test \
  --output-dir outputs/zero_shot \
  --config configs/evaluation/downstream_openai_compatible.yaml \
  --model-slug your_model
```

Score FIRAC-rendered summaries against reference summaries:

```bash
uv run firac-score-summaries \
  --predictions data/fixtures/annotations.jsonl \
  --output outputs/metrics/summarization.json \
  --split test
```

Run LLM-as-judge rubric scoring:

```bash
OPENAI_BASE_URL=https://api.example.com/v1 \
OPENAI_API_KEY=... \
FIRAC_MODEL=your-judge-model \
uv run firac-run-rubric \
  --input data/fixtures/annotations.jsonl \
  --output-dir outputs/rubric \
  --config configs/evaluation/rubric_openai_compatible.yaml \
  --prompt prompts/rubric/v1/firac_rubric_judge.md
```

Compare human Label Studio rubric exports to LLM rubric JSONL:

```bash
uv run firac-compare-rubric \
  --human-label-studio-json data/fixtures/label_studio_export.min.json \
  --llm-rubric-jsonl outputs/rubric/final.jsonl \
  --output-json outputs/metrics/human_llm_agreement.json \
  --output-csv outputs/metrics/human_llm_pairs.csv
```

Assemble compact result tables from metric JSON files:

```bash
uv run firac-make-results \
  --metrics-dir outputs/metrics \
  --output-dir outputs/tables
```

## Data

This GitHub repository carries only tiny fixtures for smoke tests. Full
reproduction should use the public FIRAC Appeals dataset release and/or the
upstream source datasets described in the paper. `firac-load-source` covers
common local IN-Abs and vanga Supreme Court layouts, but it does not download
the corpora for you. Point the commands above at downloaded/extracted source
files, annotations, task files, predictions, or metric JSON files to recreate
the paper-facing tables.

## Main Files

- `prompts/annotation/v2/`: FIRAC annotation prompts.
- `prompts/downstream/v1/`: downstream task prompts.
- `prompts/rubric/v1/`: human/LLM rubric prompts.
- `configs/annotation/`: mock and OpenAI-compatible annotation configs.
- `configs/evaluation/`: downstream, summarization, and rubric configs.
- `data/fixtures/`: small examples for local tests and documentation commands.
