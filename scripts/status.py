"""Phase 2 placeholder for experiment status inspection."""

from __future__ import annotations


def main() -> int:
    """Print a safe placeholder message for the not-yet-implemented status command."""
    print(
        "status.py is a Phase 2 placeholder.\n"
        "TODO:\n"
        "- inspect run folders and registry consistency\n"
        "- summarize planned/running/completed experiments\n"
        "- surface missing artifacts such as metrics.json or summary.md\n"
        "- optionally detect stale tmux sessions or orphaned outputs\n"
        "No actions were taken."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
