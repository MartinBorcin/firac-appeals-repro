# Rule Identifier Provenance Audit

## Scope

This audit checks whether generated precedent `case_name` and `citation` values are explicitly present
in the source judgment. It does not modify the released annotations. Source absence is only a
candidate-generation signal, not a correction criterion: normalized identifiers and canonical
enrichment are acceptable when the citation legally resolves to the named case.

- 45,016 judgments screened (7,126 IN-Abs; 37,890 Supreme Court temporal corpus)
- 85,829 precedent Rule records screened
- 42,116 Rule records selected as conservative deterministic review candidates (49.07%)
- 247 records independently adjudicated with GPT-OSS-120B using a strict source-only rubric
- Qwen-3.6-27B adjudication retained only as a cross-family diagnostic, not independent validation

The 49.07% candidate rate must not be reported as a hallucination rate. OCR, punctuation, line breaks,
short forms, and case-name variants cause many deterministic misses.

## Sample Findings

Among deterministic candidate fields, GPT-OSS-120B returned:

| Field | Candidate fields | Absent | Source variant | Exact | Uncertain |
|---|---:|---:|---:|---:|---:|
| Case name | 180 | 30 | 107 | 9 | 34 |
| Citation | 127 | 66 | 33 | 7 | 21 |

After weighting the stratified sample back to candidate strata, the estimated source-absent counts are:

- Case names: approximately 4,023 of 26,274 deterministic candidate fields
- Citations: approximately 10,066 of 25,151 deterministic candidate fields

Relative to all populated precedent fields, these correspond to approximately 4.7% of case names and
12.2% of citations being source-absent. They are provenance estimates, not estimates of legal falsity.

The wide summed stratum-level Wilson bounds in `adjudication_analysis_v2/summary.json` reflect the
small per-stratum samples and should be reported. These are model-adjudicated estimates, not expert
ground truth.

## Cross-Document Corroboration

The same generated case-name/citation pair was searched among Rule pairs for which both identifiers
are explicitly present in another raw judgment:

- 7,119 of 41,731 case/citation candidate Rules are corroborated elsewhere in the corpus.
- 2,592 of 15,450 Rules with a present case name but source-absent citation are corroborated elsewhere.
- Corroboration supports a canonical-enrichment interpretation, but does not make the value a verbatim
  extraction from the focal judgment or establish authoritative legal correctness.
- No corroboration is not proof that the identifier is wrong; aliases and OCR can prevent matching.

## Annotator-Flagged Example

For `in_abs_test_415::rule::4`, the focal judgment explicitly names *State of Bombay v. Atma Ram
Sridhar Vaidya* but does not contain the generated citation `AIR 1951 SC 184`. Deterministic search,
GPT-OSS-120B, and Qwen-3.6-27B agree that the citation is absent, and the pair is not corroborated in
the corpus. External checking shows that the correct canonical citation is `AIR 1951 SC 157` / `1951
SCR 167`. Under a strict extraction schema, the proposed focal-record patch remains to clear the
citation; under a dual schema, `AIR 1951 SC 157` may be stored separately as externally verified
canonical metadata.

External verification: https://indiankanoon.org/doc/1382411/

## Recommended Verification And Release Policy

1. Treat source-absent identifiers as a verification queue, not as errors.
2. Resolve each candidate citation against an authoritative or established legal index and compare the
   returned case title with the generated case name, allowing documented abbreviation and OCR
   normalization.
3. Label a pair `verified` when the citation resolves to the named case, `contradicted` when it resolves
   to a different case, and `unresolved` when coverage or matching is insufficient.
4. Correct only `contradicted` pairs. Retain verified canonical enrichment and publish unresolved IDs as
   review flags rather than silently deleting them.
5. Preserve lookup provenance: service, query, retrieval date, returned document ID/title, equivalent
   citations, normalization rule, and confidence.
6. Report verification coverage and error rate with confidence intervals. Do not call deterministic or
   source-absence candidates hallucinations.
7. Re-run authority extraction/recommendation evaluation after applying only verified corrections.

## Artifacts

- `rule_identifier_audit.jsonl`: full field-level deterministic audit
- `potential_issues.jsonl`: conservative review queue
- `proposed_patches.jsonl`: full deterministic, pending-review patch queue
- `adjudication_sample.jsonl`: stratified sample
- `adjudication_gpt_oss_120b_v2.jsonl`: strict independent-family adjudication
- `corpus_pair_corroboration.jsonl`: cross-document pair evidence
- `corpus_pair_corroboration_summary.json`: corroboration summary
- `adjudication_analysis_v2/review_decisions.jsonl`: one decision per sampled candidate field
- `adjudication_analysis_v2/candidate_patches.jsonl`: proposed, unapplied sample patches
- `adjudication_analysis_v2/summary.json`: weighted estimates and audit summary
