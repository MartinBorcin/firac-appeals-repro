# Strict Source-Grounded Authority Patch

## Purpose

The optional strict authority view retains precedent identifiers only when an automated verifier can
ground them in the source judgment, allowing punctuation, shorthand, abbreviation, and OCR variants.
It does not assess whether excluded canonical enrichment is legally correct.

The original dataset remains unchanged. The strict view is produced by applying a versioned patch.

## Field Semantics

- `reference` is the generic human-readable authority label used by every Rule type. For a precedent,
  it normally repeats or abbreviates `case_name`.
- `case_name` is the structured precedent name and is normally empty for non-precedent Rules.
- `citation` is the reporter, docket, or other formal case identifier and is normally used only for
  precedents.
- `provision` stores section, article, rule, order, regulation, or clause identifiers for legislation
  and similar authorities.

## Verification

The deterministic provenance screen sends a precedent Rule to GPT-OSS-120B when at least one populated
`case_name` or `citation` cannot be located conservatively in its judgment. All affected Rules from the
same judgment are verified together.

For `reference`, `case_name`, and `citation`, the verifier chooses:

- `keep`: the supplied value is supported, including cosmetic, shorthand, abbreviation, or OCR variants;
- `remove`: the value is not explicitly supported;
- `correct`: the value is materially misrepresented and the judgment supports a corrected value.

For `keep` and `correct`, the verifier returns an inclusive source-line range. The pipeline copies those
lines directly from the judgment into the audit ledger. Multi-digit citation numbers must occur in the
selected evidence. Invalid responses are retried once; remaining failures are marked `unresolved` and
produce no automatic patch operation.

## Artifacts

### `strict_authorities_audit.jsonl`

The complete verifier ledger. It contains one row for every candidate precedent Rule, including kept,
removed, corrected, and unresolved decisions; exact source evidence; validation errors; retry count;
and any proposed operations. This artifact supports paper statistics, error analysis, debugging, and
reproduction. Applying the strict view does not require reading it.

### `strict_authorities_patch.jsonl`

The user-facing actionable subset. It contains only verified Rules for which one or more identifier
values change. Operations modify only `reference`, `case_name`, or `citation`; all substantive Rule and
FIRAC fields remain untouched.

## Paper Reporting

Report the candidate and verified Rule counts, unresolved count, and kept/removed/corrected decisions
by identifier field and source corpus. Describe this as a conservative source-grounded view, not an
external legal-correctness audit. External authority linking and canonical verification remain future
work.
