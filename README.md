# FIRAC-Appeals Reproducibility Package

This repository accompanies **FIRAC-Appeals**, a structured dataset of Indian
appellate judgments. It supports four distinct uses:

1. running the FIRAC annotation pipeline on new judgments;
2. constructing and running the paper's downstream experiments;
3. rescoring the exact predictions and human/LLM assessments used in the paper;
4. preserving prompts, configurations, sanitized responses, metrics, and
   provenance needed to audit the reported results.

<img width="3269" height="861" alt="FIRAC dataset construction and task overview" src="https://github.com/user-attachments/assets/eb0971ad-6275-44b1-a46f-53aeb0e75a72" />

The full dataset is distributed separately through the anonymized dataset link
in the paper. For the camera-ready release, replace `DATASET_REPOSITORY_URL` in
this repository with the permanent dataset URL.
This code repository deliberately excludes model weights, raw upstream corpora,
judgment-bearing Label Studio exports, credentials, request databases, caches,
and cluster logs.

## Reviewer Quick Start

```bash
uv sync --extra eval
uv run pytest

# Credential-free end-to-end annotation smoke test.
uv run firac-annotate \
  --input data/fixtures/cases.jsonl \
  --output-dir outputs/mock_annotation \
  --config configs/annotation/mock.yaml

# Regenerate the human-evaluation report from released responses.
uv run python scripts/report_human_eval_round2.py \
  --input artifacts/human/label_studio_round2_sanitized.json \
  --assignment-manifest artifacts/human/assignment_manifest_sanitized.json \
  --output-dir outputs/human_report \
  --bootstrap-replicates 2000 \
  --strict-final

# Verify every released artifact checksum.
uv run python scripts/verify_artifact_manifest.py
```

## Repository Map

| Path | Purpose |
|---|---|
| `src/firac_appeals/` | Shared annotation, schema, scoring, and audit implementation |
| `scripts/` | Exact paper pipelines and standalone experiment commands |
| `prompts/` | Versioned annotation, downstream, and rubric prompts |
| `configs/` | Generic examples and exact model/evaluation configurations |
| `artifacts/` | Sanitized responses, predictions, metrics, patch, and checksums |
| `data/fixtures/` | Tiny public inputs for smoke tests |
| `tests/` | Unit and workflow tests |

## Reproduction Paths

- [Annotation pipeline](docs/ANNOTATION.md)
- [Downstream experiments](docs/DOWNSTREAM.md)
- [Human and LLM quality evaluation](docs/QUALITY_EVALUATION.md)
- [Source-grounded authority patch](docs/AUTHORITY_PATCH.md)
- [Artifact inventory and release policy](docs/ARTIFACTS.md)
- [Anonymous review release checklist](docs/REVIEW_RELEASE.md)
- [Exact paper results and provenance](docs/RESULTS.md)

## Installation

Python 3.12 and [`uv`](https://docs.astral.sh/uv/) are recommended.

```bash
uv sync --extra eval
```

Training InLegalBERT additionally requires a CUDA-compatible PyTorch setup:

```bash
uv sync --extra eval --extra train
```

LLM runs require an OpenAI-compatible chat-completions endpoint and a user
supplied API key. Configurations reference environment-variable names only; no
credentials are included.

## Annotation Example

Normalize local JSONL, JSON, or CSV judgments:

```bash
uv run firac-prepare-cases \
  --input my_cases.jsonl \
  --output outputs/cases.jsonl \
  --text-field judgment_text \
  --id-field case_id
```

Then run the same staged FIRAC pipeline against an OpenAI-compatible endpoint:

```bash
OPENAI_BASE_URL=https://your-endpoint.example/v1 \
OPENAI_API_KEY=... \
FIRAC_MODEL=your-model \
uv run firac-annotate \
  --input outputs/cases.jsonl \
  --output-dir outputs/annotations \
  --config configs/annotation/firac_pipeline.yaml
```

The prompts used for the released annotations are under
`prompts/annotation/v2/`. See [docs/ANNOTATION.md](docs/ANNOTATION.md) for stage
semantics, schemas, resumption, and expected inputs.

## What Is Reproducible Without Paid Services?

- All metric computation, task construction, retrieval baselines, TF-IDF+LR,
  human statistics, authority screening, and artifact verification.
- InLegalBERT training when suitable GPU compute and model access are supplied.
- Annotation and zero-shot/judge inference when the named models or compatible
  endpoints are supplied.
- Exact rescoring from shipped predictions without rerunning LLM inference.

See [docs/ARTIFACTS.md](docs/ARTIFACTS.md) for deliberate exclusions and the
relationship between this repository and the dataset release.

## Release Verification

```bash
uv run pytest
uv run ruff check .
uv run python scripts/build_artifact_manifest.py
uv run python scripts/verify_artifact_manifest.py

# These searches should return no matches in an anonymous release.
rg -n '(hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|BEGIN .*PRIVATE KEY)' .
rg -n -i '(hf\.space|/users/|/home/|\.ac\.uk)' \
  --glob '!README.md' --glob '!artifacts/MANIFEST.json'
```

The text scans are safeguards rather than credential detectors; inspect any
matches before publishing. Git history is handled separately as described in
the anonymous review checklist.
