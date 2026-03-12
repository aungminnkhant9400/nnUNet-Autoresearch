"""Phase 2 placeholder for collecting metrics into experiment metadata."""

from __future__ import annotations


def main() -> int:
    """Print a safe placeholder message instead of mutating metrics."""
    print(
        "collect_metrics.py is a Phase 2 placeholder.\n"
        "TODO:\n"
        "- parse nnU-Net output metrics from JSON, CSV, or text summaries\n"
        "- normalize Dice, HD95, precision, and recall fields\n"
        "- update metrics.json and the results section of meta.yaml\n"
        "- rerun registry synchronization after successful collection\n"
        "This placeholder does not change any files."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
