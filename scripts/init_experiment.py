"""Initialize a new nnU-Net autoresearch experiment folder."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

DEFAULT_NNUNET_VERSION = "v2"
DEFAULT_PRIMARY_METRIC = "dice"
DEFAULT_TRAINER = "nnUNetTrainer"
DEFAULT_CONFIGURATION = "3d_fullres"


def project_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file as a dictionary."""
    if not path.exists():
        raise FileNotFoundError(f"Missing YAML file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML file: {path}")
    return data


def save_yaml(path: Path, payload: dict[str, Any]) -> None:
    """Write a dictionary to YAML with stable formatting."""
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create a new experiment folder with metadata placeholders."
    )
    parser.add_argument("--dataset", required=True, help="Dataset key, for example psma5mm.")
    parser.add_argument(
        "--type",
        required=True,
        choices=["training", "inference", "postprocess"],
        dest="task_type",
        help="Experiment class.",
    )
    parser.add_argument("--title", required=True, help="Human-readable experiment title.")
    parser.add_argument("--fold", help="Fold identifier such as 0, 1, or all.")
    parser.add_argument(
        "--trainer",
        default=DEFAULT_TRAINER,
        help=f"Trainer name. Default: {DEFAULT_TRAINER}.",
    )
    parser.add_argument(
        "--configuration",
        default=DEFAULT_CONFIGURATION,
        help=f"nnU-Net configuration. Default: {DEFAULT_CONFIGURATION}.",
    )
    parser.add_argument("--checkpoint", default="", help="Checkpoint reference if applicable.")
    parser.add_argument("--parent", default="", help="Parent experiment id, if any.")
    parser.add_argument("--objective", default="", help="Primary experiment objective.")
    parser.add_argument("--hypothesis", default="", help="One-line experiment hypothesis.")
    parser.add_argument(
        "--change-type",
        default="baseline",
        help="Short tag describing the main intervention.",
    )
    parser.add_argument(
        "--change-details",
        default="",
        help="Longer description of the change being tested.",
    )
    parser.add_argument("--device", default="", help="Execution device such as cuda:0.")
    return parser.parse_args()


def discover_next_experiment_number(root: Path) -> int:
    """Find the next sequential experiment number from runs and registry."""
    numbers: list[int] = []
    pattern = re.compile(r"^exp_(\d{4})")

    runs_dir = root / "runs"
    if runs_dir.exists():
        for child in runs_dir.iterdir():
            if not child.is_dir():
                continue
            match = pattern.match(child.name)
            if match:
                numbers.append(int(match.group(1)))

    registry_path = root / "registry" / "experiments.jsonl"
    if registry_path.exists():
        with registry_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                experiment_id = str(record.get("experiment_id", ""))
                match = pattern.match(experiment_id)
                if match:
                    numbers.append(int(match.group(1)))

    return (max(numbers) if numbers else 0) + 1


def slugify(value: str) -> str:
    """Convert a title into a filesystem-friendly slug."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "experiment"


def resolve_dataset(root: Path, dataset_key: str) -> dict[str, Any]:
    """Resolve dataset configuration by filename or embedded dataset key."""
    datasets_dir = root / "config" / "datasets"
    candidate = datasets_dir / f"{dataset_key}.yaml"
    if candidate.exists():
        return load_yaml(candidate)

    for path in sorted(datasets_dir.glob("*.yaml")):
        payload = load_yaml(path)
        if payload.get("dataset_key") == dataset_key:
            return payload

    raise FileNotFoundError(
        f"Could not resolve dataset '{dataset_key}' in {datasets_dir}"
    )


def load_defaults(root: Path) -> dict[str, Any]:
    """Load global configuration defaults."""
    return load_yaml(root / "config" / "global.yaml")


def load_experiment_types(root: Path) -> dict[str, Any]:
    """Load experiment type configuration."""
    return load_yaml(root / "config" / "experiment_types.yaml")


def validate_task_requirements(
    args: argparse.Namespace, experiment_types: dict[str, Any]
) -> None:
    """Validate task-type-specific required CLI fields."""
    task_config = experiment_types.get(args.task_type, {})
    if not isinstance(task_config, dict):
        raise ValueError(f"Invalid experiment type configuration for {args.task_type}")

    required_inputs = task_config.get("required_inputs", [])
    if not isinstance(required_inputs, list):
        raise ValueError(
            f"Invalid required_inputs for experiment type {args.task_type}"
        )

    value_by_field = {
        "checkpoint": args.checkpoint,
        "parent": args.parent,
    }
    flag_by_field = {
        "checkpoint": "--checkpoint",
        "parent": "--parent",
    }

    missing_flags = [
        flag_by_field[field]
        for field in required_inputs
        if not str(value_by_field.get(field, "")).strip()
    ]
    if missing_flags:
        raise ValueError(
            f"{args.task_type} experiments require: {', '.join(missing_flags)}"
        )


def default_slug(args: argparse.Namespace) -> str:
    """Build a default folder slug."""
    fold = f"fold{args.fold}" if args.fold not in (None, "") else ""
    change_tag = slugify(args.change_type) if args.change_type else "baseline"

    if args.task_type == "training":
        parts = ["baseline", fold, slugify(args.configuration)]
    elif args.task_type == "inference":
        checkpoint = slugify(args.checkpoint) if args.checkpoint else "best"
        parts = ["predict", fold, checkpoint]
    else:
        post_tag = change_tag if change_tag != "baseline" else "default"
        parts = [fold, "post", post_tag]

    return "_".join(part for part in parts if part)


def build_meta(
    args: argparse.Namespace,
    experiment_id: str,
    run_dir: Path,
    dataset_config: dict[str, Any],
    global_config: dict[str, Any],
    command_text: str,
) -> dict[str, Any]:
    """Create the initial experiment metadata payload."""
    primary_metric = (
        dataset_config.get("evaluation", {}).get("primary_metric") or DEFAULT_PRIMARY_METRIC
    )
    device = args.device or str(global_config.get("default_device", "cuda:0"))

    return {
        "experiment_id": experiment_id,
        "title": args.title,
        "dataset_key": dataset_config.get("dataset_key", args.dataset),
        "nnunet_version": DEFAULT_NNUNET_VERSION,
        "task_type": args.task_type,
        "status": "planned",
        "parent_experiment": args.parent,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "goal": {
            "primary_metric": primary_metric,
            "objective": args.objective,
            "hypothesis": args.hypothesis,
        },
        "inputs": {
            "trainer": args.trainer,
            "configuration": args.configuration,
            "fold": args.fold or "",
            "checkpoint": args.checkpoint,
        },
        "change": {
            "type": args.change_type,
            "details": args.change_details,
        },
        "execution": {
            "device": device,
            "tmux_session": "",
            "command": command_text,
        },
        "results": {
            "dice_mean": None,
            "dice_median": None,
            "hd95_mean": None,
            "precision_mean": None,
            "recall_mean": None,
        },
        "decision": {
            "verdict": "pending",
            "rationale": "",
        },
        "notes": [],
        "run": {
            "path": str(run_dir.relative_to(project_root())).replace("\\", "/"),
        },
    }


def command_template(meta: dict[str, Any]) -> str:
    """Build a safe placeholder command script."""
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"# Experiment: {meta['experiment_id']} - {meta['title']}",
        f"# Dataset: {meta['dataset_key']}",
        f"# Type: {meta['task_type']}",
        "# Review and replace this placeholder before running anything.",
        "",
    ]

    if meta["task_type"] == "training":
        lines.extend(
            [
                "# Example training command:",
                "# nnUNetv2_train <DATASET_ID> all "
                f"{meta['inputs']['configuration']} {meta['inputs']['trainer']}",
            ]
        )
    elif meta["task_type"] == "inference":
        lines.extend(
            [
                "# Example inference command:",
                "# nnUNetv2_predict -i <INPUT_DIR> -o <OUTPUT_DIR> -d <DATASET_ID> "
                f"-c {meta['inputs']['configuration']}",
            ]
        )
    else:
        lines.extend(
            [
                "# Example postprocess command:",
                "# python custom_postprocess.py --input <PRED_DIR> --output <POST_DIR>",
            ]
        )

    return "\n".join(lines) + "\n"


def summary_template(meta: dict[str, Any]) -> str:
    """Build a human-editable Markdown summary scaffold."""
    return (
        f"# {meta['experiment_id']} - {meta['title']}\n\n"
        f"- Dataset: `{meta['dataset_key']}`\n"
        f"- Type: `{meta['task_type']}`\n"
        f"- Status: `{meta['status']}`\n"
        f"- Parent: `{meta['parent_experiment'] or 'none'}`\n"
        f"- Created: `{meta['created_at']}`\n"
        f"- Device: `{meta['execution']['device']}`\n"
        f"- Primary Metric: `{meta['goal']['primary_metric']}`\n\n"
        "## Objective\n\n"
        f"{meta['goal']['objective'] or 'TODO'}\n\n"
        "## Hypothesis\n\n"
        f"{meta['goal']['hypothesis'] or 'TODO'}\n\n"
        "## Planned Command Review\n\n"
        "Review `command.sh` and fill in the exact nnU-Net invocation before launch.\n\n"
        "## Notes\n\n"
        "- Add run notes here.\n"
    )


def write_text(path: Path, content: str) -> None:
    """Write text content to a file."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def reserve_run_directory(root: Path, slug: str) -> tuple[str, Path]:
    """Reserve a unique experiment directory using atomic directory creation."""
    runs_dir = root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    for experiment_number in range(1, 10000):
        experiment_id = f"exp_{experiment_number:04d}"
        reservation_dir = runs_dir / f".{experiment_id}.reserve"
        run_dir = runs_dir / f"{experiment_id}_{slug}"

        try:
            reservation_dir.mkdir()
        except FileExistsError:
            continue

        try:
            if any(runs_dir.glob(f"{experiment_id}_*")):
                reservation_dir.rmdir()
                continue

            reservation_dir.replace(run_dir)
            return experiment_id, run_dir
        except Exception:
            if reservation_dir.exists():
                reservation_dir.rmdir()
            raise

    raise RuntimeError("Could not reserve a new experiment id under runs/")


def main() -> int:
    """Create the experiment folder and placeholder files."""
    args = parse_args()
    root = project_root()
    dataset_config = resolve_dataset(root, args.dataset)
    global_config = load_defaults(root)
    experiment_types = load_experiment_types(root)
    validate_task_requirements(args, experiment_types)

    title_slug = slugify(args.title)
    slug = title_slug if title_slug else default_slug(args)
    if args.title.strip().lower() in {"", "default"}:
        slug = default_slug(args)

    experiment_id, run_dir = reserve_run_directory(root, slug)
    meta = build_meta(
        args,
        experiment_id,
        run_dir,
        dataset_config,
        global_config,
        command_text="",
    )
    command_text = command_template(meta)
    meta["execution"]["command"] = command_text

    save_yaml(run_dir / "meta.yaml", meta)
    write_text(run_dir / "command.sh", command_text)
    write_text(run_dir / "summary.md", summary_template(meta))
    write_text(run_dir / "metrics.json", "{}\n")

    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
