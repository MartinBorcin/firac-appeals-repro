# Annotation Pipeline

## Released configuration

The released FIRAC records were generated with Qwen3.6-27B, thinking disabled,
and the prompts in `prompts/annotation/v2/`. Each judgment passes through four
specialized roles:

1. `issue_analyst` identifies dispositive legal questions;
2. `legal_researcher` extracts Rules and Authorities;
3. `logician` reconstructs issue-organized rule application;
4. `firac_judge` reconciles the intermediate outputs into validated FIRAC JSON.

Facts are instructed to contain pre-decision factual and procedural material,
excluding the present appellate court's reasoning and outcome. The generic,
portable configuration is `configs/annotation/firac_pipeline.yaml`; the exact
high-throughput vLLM configuration is `configs/annotation/local_vllm_qwen.yaml`.

## Input contract

Annotation consumes JSONL with at least:

```json
{"case_id":"stable-id","judgment_text":"Full judgment text","split":"train"}
```

Additional source metadata is preserved. Use `firac-prepare-cases` for arbitrary
JSONL/JSON/CSV, or `firac-load-source` for the supported IN-Abs and Supreme Court
layouts.

The exact Supreme Court source download/normalization procedure is retained in
`scripts/prepare_supreme_court_dataset.py`; temporal splits are created with
`firac-split-temporal`.

## Run

```bash
OPENAI_BASE_URL=https://your-endpoint.example/v1 \
OPENAI_API_KEY=... \
FIRAC_MODEL=your-model \
uv run firac-annotate \
  --input cases.jsonl \
  --output-dir outputs/annotations \
  --config configs/annotation/firac_pipeline.yaml
```

`final.jsonl` is the portable result. SQLite request stores and per-request logs
are runtime aids and are not release artifacts. Runs are resumable when
`output.skip_done` is enabled.

## Reusing the pipeline

Prompts and stage token limits may be changed through YAML without modifying
code. Any OpenAI-compatible endpoint can be used, but model-specific chat
templates and structured-output behavior remain the responsibility of the
runner. The mock backend and `data/fixtures/cases.jsonl` provide a
credential-free integration test.
