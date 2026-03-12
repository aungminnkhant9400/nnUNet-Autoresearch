"""Phase 2 placeholder for launching nnU-Net training runs."""

from __future__ import annotations


def main() -> int:
    """Print a safe placeholder message instead of launching training."""
    print(
        "launch_train.py is a Phase 2 placeholder.\n"
        "TODO:\n"
        "- load meta.yaml and validate required training fields\n"
        "- verify GPU availability and policy constraints before launch\n"
        "- render a reviewed nnUNetv2_train command from metadata\n"
        "- support tmux session creation without overwriting previous outputs\n"
        "This placeholder does not start any process."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
