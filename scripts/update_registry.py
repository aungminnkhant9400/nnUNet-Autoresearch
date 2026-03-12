"""Update experiment registry files from a run's meta.yaml."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

LEADERBOARD_COLUMNS = [
    "experiment_id",
    "title",
    "dataset_key",
    "task_type",
    "status",
    "created_at",
    "parent_experiment",
    "primary_metric",
    "dice_mean",
    "dice_median",
    "hd95_mean",
    "precision_mean",
    "recall_mean",
    "verdict",
    "trainer",
    "configuration",
    "fold",
    "checkpoint",
    "device",
    "run_path",
]


def project_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Update experiments.jsonl and leaderboard.csv from meta.yaml."
    )
    parser.add_argument(
        "--meta",
        required=True,
        help="Path to a run meta.yaml file or its parent run directory.",
    )
    return parser.parse_args()


def normalize_meta_path(raw_path: str) -> Path:
    """Resolve a meta.yaml path from either a file path or run directory."""
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = project_root() / path
    if path.is_dir():
        path = path / "meta.yaml"
    return path.resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    """Load and validate a YAML mapping."""
    if not path.exists():
        raise FileNotFoundError(f"Missing meta.yaml: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records, skipping blank lines."""
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL entry in {path} on line {line_number}: {exc}"
                ) from exc
            if isinstance(record, dict):
                records.append(record)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write JSONL records in a stable order."""
    content = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n"
        for record in records
    )
    atomic_write_text(path, content, newline="\n")


def atomic_write_text(path: Path, content: str, newline: str | None = None) -> None:
    """Write a file atomically using a temp file in the same directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temp_path = Path(raw_temp_path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline=newline) as handle:
            handle.write(content)
        os.replace(temp_path, path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def nested_get(payload: dict[str, Any], *keys: str) -> Any:
    """Safely access nested dictionary values."""
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return ""
        current = current.get(key, "")
    return current


def normalize_cell(value: Any) -> Any:
    """Normalize missing values to empty strings for CSV output."""
    if value is None:
        return ""
    if isinstance(value, list):
        return " | ".join(str(item) for item in value)
    return value


def experiment_sort_key(record: dict[str, Any]) -> tuple[int, str]:
    """Sort experiment records by numeric id then lexicographically."""
    experiment_id = str(record.get("experiment_id", ""))
    try:
        number = int(experiment_id.split("_")[1])
    except (IndexError, ValueError):
        number = 0
    return number, experiment_id


def leaderboard_row(meta: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested metadata for leaderboard.csv."""
    row = {
        "experiment_id": meta.get("experiment_id", ""),
        "title": meta.get("title", ""),
        "dataset_key": meta.get("dataset_key", ""),
        "task_type": meta.get("task_type", ""),
        "status": meta.get("status", ""),
        "created_at": meta.get("created_at", ""),
        "parent_experiment": meta.get("parent_experiment", ""),
        "primary_metric": nested_get(meta, "goal", "primary_metric"),
        "dice_mean": nested_get(meta, "results", "dice_mean"),
        "dice_median": nested_get(meta, "results", "dice_median"),
        "hd95_mean": nested_get(meta, "results", "hd95_mean"),
        "precision_mean": nested_get(meta, "results", "precision_mean"),
        "recall_mean": nested_get(meta, "results", "recall_mean"),
        "verdict": nested_get(meta, "decision", "verdict"),
        "trainer": nested_get(meta, "inputs", "trainer"),
        "configuration": nested_get(meta, "inputs", "configuration"),
        "fold": nested_get(meta, "inputs", "fold"),
        "checkpoint": nested_get(meta, "inputs", "checkpoint"),
        "device": nested_get(meta, "execution", "device"),
        "run_path": nested_get(meta, "run", "path"),
    }
    return {column: normalize_cell(row.get(column, "")) for column in LEADERBOARD_COLUMNS}


def write_leaderboard(path: Path, records: list[dict[str, Any]]) -> None:
    """Regenerate leaderboard.csv from the full registry."""
    rows = [leaderboard_row(record) for record in records]
    rows.sort(
        key=lambda row: (
            row.get("dataset_key", ""),
            row.get("task_type", ""),
            row.get("experiment_id", ""),
        )
    )

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=LEADERBOARD_COLUMNS)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    atomic_write_text(path, buffer.getvalue(), newline="")


def validate_meta(meta: dict[str, Any]) -> None:
    """Validate required top-level metadata fields."""
    required = [
        "experiment_id",
        "title",
        "dataset_key",
        "nnunet_version",
        "task_type",
        "status",
        "parent_experiment",
        "created_at",
        "goal",
        "inputs",
        "change",
        "execution",
        "results",
        "decision",
        "notes",
    ]
    missing = [key for key in required if key not in meta]
    if missing:
        raise ValueError(f"meta.yaml is missing required keys: {', '.join(missing)}")

    required_sections = ["goal", "inputs", "change", "execution", "results", "decision"]
    for section_name in required_sections:
        section = meta.get(section_name)
        if not isinstance(section, dict):
            raise ValueError(f"meta.yaml section '{section_name}' must be a mapping")

    nested_required_keys = {
        "goal": ["primary_metric", "objective"],
        "inputs": ["configuration"],
        "decision": ["verdict"],
    }
    for section_name, keys in nested_required_keys.items():
        section = meta[section_name]
        missing_keys = [key for key in keys if key not in section]
        if missing_keys:
            raise ValueError(
                f"meta.yaml section '{section_name}' is missing keys: {', '.join(missing_keys)}"
            )


def upsert_record(
    records: list[dict[str, Any]], candidate: dict[str, Any]
) -> list[dict[str, Any]]:
    """Replace or append a record keyed by experiment_id."""
    experiment_id = str(candidate.get("experiment_id", "")).strip()
    if not experiment_id:
        raise ValueError("meta.yaml must include a non-empty experiment_id")

    by_id: dict[str, dict[str, Any]] = {
        str(record.get("experiment_id", "")).strip(): record for record in records
    }
    by_id[experiment_id] = candidate
    return [
        record
        for _, record in sorted(
            by_id.items(),
            key=lambda item: experiment_sort_key(item[1]),
        )
    ]


def main() -> int:
    """Update the JSONL registry and derived leaderboard CSV."""
    args = parse_args()
    meta_path = normalize_meta_path(args.meta)
    meta = load_yaml(meta_path)
    validate_meta(meta)

    root = project_root()
    registry_dir = root / "registry"
    jsonl_path = registry_dir / "experiments.jsonl"
    leaderboard_path = registry_dir / "leaderboard.csv"

    records = load_jsonl(jsonl_path)
    records = upsert_record(records, meta)

    write_jsonl(jsonl_path, records)
    write_leaderboard(leaderboard_path, records)

    print(f"Updated registry for {meta.get('experiment_id', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
