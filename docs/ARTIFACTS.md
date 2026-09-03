# Artifact Inventory and Release Policy

`artifacts/MANIFEST.json` records the byte size and SHA-256 digest of every
released artifact. Run `scripts/verify_artifact_manifest.py` after cloning.

## Included

### Human evaluation

- privacy-preserving Label Studio response export;
- pseudonymized assignment manifest;
- criterion-level, case-level, agreement, and low-score outputs;
- final human/LLM tables and provenance metadata.

The sanitized export is the earliest public input needed to regenerate the
statistics. It contains no judgment text, account identifiers, or real email
addresses; annotators are represented by stable aliases under the reserved
`.invalid` domain.

### LLM quality screen

- complete GPT-OSS-120B JSONL outputs for IN-Abs and Supreme Court;
- derived Likert and alignment tables under `artifacts/human/`.

These outputs are retained because API/model behavior may change and rerunning
45,000 judgments is expensive.

### Downstream experiments

- final issue predictions and common MiniLM rescoring metrics;
- authority retrieval/generation predictions and metrics;
- binary LJP predictions, TF-IDF coefficients, three-seed InLegalBERT results,
  and overlap/signal audits;
- summarization predictions and per-case/aggregate metrics.

References and task inputs are distributed with the FIRAC-Appeals dataset rather
than duplicated here.

### Authority provenance

- actionable strict patch and compact counts;
- deterministic audit summary and cross-document corroboration summary;
- one externally checked correction record.

The full 852 MB evidence ledger is omitted from Git because it repeats long
source passages and is not needed to apply the patch. The actionable patch,
method, counts, and code required to regenerate the ledger are included.

## Deliberately excluded

| Excluded material | Reason |
|---|---|
| Raw upstream judgments | Distributed by upstream sources/full dataset |
| Unsanitized Label Studio export | Contains judgments and account identifiers |
| `.env`, API keys, passwords | Credentials are user supplied |
| SQLite request stores and logs | Operational, large, and may contain prompts/text |
| Embedding caches | Deterministically regenerable and hundreds of MB |
| InLegalBERT checkpoints | Multi-GB; reproducible from the public base model |
| Failed, smoke, and debug runs | Not paper evidence |
| Cluster-specific Slurm output | Operational rather than scientific provenance |

Model availability, endpoint access, and compute are external prerequisites.
The repository records model identifiers, prompts, sampling parameters, seeds,
and training settings but cannot redistribute third-party models or services.
