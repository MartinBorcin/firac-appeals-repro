from __future__ import annotations

import argparse

from firac_appeals.ingest.in_abs import load_in_abs
from firac_appeals.ingest.vanga import load_vanga_supreme_court
from firac_appeals.utils.io import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Load supported raw/source datasets into FIRAC case JSONL.")
    parser.add_argument("--source", required=True, choices=["in_abs", "supreme_court_vanga"])
    parser.add_argument("--input", required=True, help="Source file or extracted source directory")
    parser.add_argument("--output", required=True, help="Annotation-ready case JSONL")
    parser.add_argument(
        "--metadata",
        default="",
        help="Optional metadata JSONL/JSON/CSV/parquet for supreme_court_vanga extracted files.",
    )
    args = parser.parse_args()

    if args.source == "in_abs":
        rows = load_in_abs(args.input)
    elif args.source == "supreme_court_vanga":
        rows = load_vanga_supreme_court(args.input, metadata_path=args.metadata or None)
    else:  # pragma: no cover - argparse enforces choices
        raise ValueError(args.source)

    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} {args.source} records to {args.output}")


if __name__ == "__main__":
    main()
