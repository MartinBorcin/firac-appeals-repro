# Source-Grounded Authority Patch

The optional patch serves researchers who require every precedent identifier to
be supported by the focal judgment. It does not replace the original enriched
dataset and does not claim external legal verification.

All 85,829 precedent Rules were screened by normalized source matching. The
41,731 Rules whose case name or citation could not be located reliably were
reviewed by GPT-OSS-120B using judgment text only. OCR variants, abbreviations,
and clear shorthand could be retained; unsupported identifiers were removed and
materially misrepresented identifiers could be corrected. Retained or corrected
values required source evidence checked by the pipeline.

Valid decisions were obtained for 24,824 candidates; 16,907 unresolved Rules
produced no changes. The actionable patch contains 26,012 field edits across
12,476 Rules: 25,979 removals and 33 corrections. Only `reference`, `case_name`,
and `citation` are changed.

Files:

- `artifacts/authority_patch/strict_authorities_patch.jsonl`: actionable patch;
- `artifacts/authority_patch/strict_patch_summary.json`: compact counts;
- `artifacts/authority_patch/summary.json` and `REPORT.md`: deterministic audit;
- `scripts/audit_rule_identifier_provenance.py`: first-stage screen;
- `scripts/run_strict_authority_verification.py`: source-only verification.

Source absence is not proof that an identifier is legally false: a valid
canonical citation may be absent from the focal judgment. Canonical resolution
against external legal databases remains separate future work.
