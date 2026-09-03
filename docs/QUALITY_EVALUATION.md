# Human and LLM Quality Evaluation

## Shared rubric

Both evaluations score Faithfulness/Correctness and Coverage/Completeness for
Facts, Issues, Rules/Authorities, Analysis, and Conclusion on a five-point
Likert scale. The exact human interface is
`configs/label_studio/firac_eval_round2.xml`. The full-corpus machine prompt is
`prompts/rubric/v4_firac_rubric_judge.md`.

The human interface requires a short explanation for every score of 3 or lower.
It does not ask annotators for machine-oriented critical flags. Such flags in
the LLM output are diagnostic categories for aggregating expected failure modes,
not an additional human rubric dimension.

## Human evaluation

`artifacts/human/label_studio_round2_sanitized.json` is the immutable public
input. It preserves task strata, pseudonymous annotators, scores, explanations,
and optional notes while removing judgments, real emails, and Label Studio
account and project metadata. No live Label Studio access is required.

`artifacts/human/assignment_manifest_sanitized.json` records the realized
stratified assignment and 30-case overlap design. The one-off operational script
used to distribute work is intentionally excluded.

The balanced, source-stratified sample can be reconstructed with
`scripts/build_label_studio_round2.py`. The exact task rendering used by the
interface is in `scripts/export_label_studio_firac_eval.py`; neither script
connects to or provisions a Label Studio instance.

```bash
uv run python scripts/report_human_eval_round2.py \
  --input artifacts/human/label_studio_round2_sanitized.json \
  --assignment-manifest artifacts/human/assignment_manifest_sanitized.json \
  --output-dir outputs/human_report \
  --bootstrap-replicates 2000 --seed 20260817 --strict-final
```

The primary quality estimate averages annotators within case and then cases.
Intervals use a case-clustered percentile bootstrap. Reliability on the 30
double-annotated cases is reported with ordinal Krippendorff's alpha, exact and
within-one agreement, and mean absolute difference.

## Full-corpus LLM screen

GPT-OSS-120B was used instead of the Qwen annotation model. The exact runtime
configuration is `configs/evaluation/llm_rubric_gpt_oss_120b_public_v4.yaml`.
The two complete output files are in `artifacts/llm_judge/`; failures remain
visible through coverage accounting. This screen is descriptive at scale and is
not presented as a substitute for expert review.

The final joint tables can be rebuilt with:

```bash
uv run python scripts/build_quality_evaluation_tables.py \
  --human-export artifacts/human/label_studio_round2_sanitized.json \
  --human-summary outputs/human_report/summary.json \
  --assignment-manifest artifacts/human/assignment_manifest_sanitized.json \
  --llm-jsonl artifacts/llm_judge/in_abs.jsonl \
  --llm-jsonl artifacts/llm_judge/supreme_court.jsonl \
  --expected-llm-cases 45020 \
  --output-dir outputs/quality_tables
```

The costly full-corpus inference itself is exposed by
`uv run python scripts/run_llm_rubric_eval.py --help`; rescoring does not require
rerunning it because the complete outputs are released.
