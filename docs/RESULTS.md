# Exact Paper Results

This page is an index, not a second source of truth. Machine-readable metrics
and predictions live under `artifacts/`; the checksum manifest fixes their exact
release versions.

## Summarization

IN-Abs test (`n=100`):

| FIRAC renderer | R-1 | R-2 | R-L | BERTScore F1 |
|---|---:|---:|---:|---:|
| Uncapped | .497 | .240 | .243 | .854 |
| Oracle length-controlled | .548 | .245 | .262 | .854 |

Artifacts: `artifacts/summarization/`.

## Issue identification

| System | IN-Abs semantic F1 | SC semantic F1 |
|---|---:|---:|
| Jaccard NN | .579 | .604 |
| MPNet NN | .598 | .609 |
| Qwen3.6-27B | .757 | .738 |
| Qwen3-32B | .721 | .704 |
| Mistral 24B | .739 | .721 |
| SaulLM-54B | .742 | .712 |

All rows use common MiniLM scoring. Full precision, recall, token F1, and mean
prediction counts are in `artifacts/downstream/issue/metrics/`.

## Authority recommendation

| System | IN-Abs strict nDCG@5 | SC strict nDCG@5 |
|---|---:|---:|
| Lexical retrieval | .035 | .020 |
| Dense retrieval | .104 | .108 |
| Qwen3.6-27B | .214 | .266 |
| Qwen3-32B | .192 | .212 |
| Mistral 24B | .293 | .232 |
| SaulLM-54B | .126 | .109 |
| Mistral, predicted Issues | .250 | .189 |
| Mistral, predicted Issues + selection | .188 | .186 |

The ranked metrics use strict normalized identifier equality and should be read
with token F1. Full outputs are in `artifacts/downstream/authority/`.

## Binary LJP

| System | IN-Abs Macro-F1 | SC Macro-F1 |
|---|---:|---:|
| Majority | .338 | .428 |
| Qwen3.6-27B | .657 | .663 |
| Mistral 24B | .557 | .631 |
| Saul-7B | .375 | .450 |
| SaulLM-54B | .449 | .453 |
| TF-IDF + LR | .663 | .629 |
| InLegalBERT (3 seeds) | .621 +/- .037 | .700 +/- .011 |

Accuracy, MCC, predictions, per-seed histories, and signal audits are in
`artifacts/downstream/ljp/`.

## Quality evaluation

The final human report contains 150 annotations over 120 cases, including 30
double-annotated cases. The case-level overall mean is 4.780 and ordinal
Krippendorff's alpha is .401. The GPT-OSS screen produced valid results for
45,003 of 45,015 attempted records; its overall criterion mean is 4.733, with
Rules/Authorities Coverage lowest at 4.207.

Canonical files:

- `artifacts/human/report/summary.json`;
- `artifacts/human/quality_tables.json`;
- `artifacts/llm_judge/in_abs.jsonl`;
- `artifacts/llm_judge/supreme_court.jsonl`.

## Authority patch

The optional patch contains 26,012 identifier-field edits over 12,476 precedent
Rules in 6,737 judgments. See `artifacts/authority_patch/strict_patch_summary.json`.
