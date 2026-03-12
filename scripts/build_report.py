"""Build a Markdown experiment report from the registry."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


def project_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a Markdown summary report from experiments.jsonl."
    )
    parser.add_argument("--dataset", default="", help="Optional dataset filter.")
    parser.add_argument(
        "--output",
        default="",
        help="Output Markdown file. Defaults to reports/report_<scope>.md.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records from disk."""
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL entry in {path} on line {line_number}: {exc}"
                ) from exc
            if isinstance(payload, dict):
                records.append(payload)
    return records


def load_global_config() -> dict[str, Any]:
    """Load global configuration if available."""
    path = project_root() / "config" / "global.yaml"
    if not path.exists():
        raise FileNotFoundError(
            "Missing runtime config: "
            f"{path}. Copy from {project_root() / 'config' / 'global.example.yaml'} "
            "and edit it for this machine."
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def nested_get(payload: dict[str, Any], *keys: str) -> Any:
    """Safely access nested dictionary values."""
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def parse_float(value: Any) -> float | None:
    """Convert a value to float if possible."""
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def escape_markdown_text(value: Any) -> str:
    """Escape Markdown table-breaking characters."""
    if value in (None, ""):
        return ""
    return str(value).replace("\r", "").replace("\n", "<br>").replace("|", "\\|")


def markdown_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    """Render a Markdown table."""
    materialized = [[escape_markdown_text(cell) for cell in row] for row in rows]
    header_line = "| " + " | ".join(escape_markdown_text(header) for header in headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = ["| " + " | ".join(row) + " |" for row in materialized]
    return "\n".join([header_line, separator_line, *row_lines])


def status_counts(records: list[dict[str, Any]]) -> list[tuple[str, int]]:
    """Count experiments by status."""
    counts: dict[str, int] = {}
    for record in records:
        status = str(record.get("status", "") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return sorted(counts.items(), key=lambda item: item[0])


def metric_result_key(metric_name: str) -> str:
    """Normalize a primary metric name to a results field key."""
    normalized = str(metric_name or "").strip()
    if not normalized:
        return "dice_mean"
    if normalized.endswith("_mean") or normalized.endswith("_median"):
        return normalized
    return f"{normalized}_mean"


def top_by_metric(
    records: list[dict[str, Any]], metric_key: str, limit: int
) -> list[dict[str, Any]]:
    """Return top experiments by the configured primary metric."""
    with_scores = []
    for record in records:
        score = parse_float(nested_get(record, "results", metric_key))
        if score is None:
            continue
        with_scores.append((score, record))
    with_scores.sort(key=lambda item: item[0], reverse=True)
    return [record for _, record in with_scores[:limit]]


def latest_records(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Return the latest experiments by created_at descending."""
    return sorted(
        records,
        key=lambda record: str(record.get("created_at", "")),
        reverse=True,
    )[:limit]


def collect_notes(records: list[dict[str, Any]], limit: int) -> list[str]:
    """Collect brief notes from the latest experiments."""
    notes: list[str] = []
    for record in latest_records(records, limit=limit * 2):
        experiment_id = str(record.get("experiment_id", ""))
        record_notes = record.get("notes", [])
        if isinstance(record_notes, str) and record_notes.strip():
            notes.append(
                f"- `{escape_markdown_text(experiment_id)}`: {escape_markdown_text(record_notes.strip())}"
            )
        elif isinstance(record_notes, list):
            for item in record_notes:
                text = str(item).strip()
                if text:
                    notes.append(
                        f"- `{escape_markdown_text(experiment_id)}`: {escape_markdown_text(text)}"
                    )
                    if len(notes) >= limit:
                        return notes

        rationale = str(nested_get(record, "decision", "rationale") or "").strip()
        if rationale:
            notes.append(
                f"- `{escape_markdown_text(experiment_id)}` decision: {escape_markdown_text(rationale)}"
            )
            if len(notes) >= limit:
                return notes
    return notes[:limit]


def default_output_path(dataset: str, output_dir: str) -> Path:
    """Build the default report output path."""
    scope = dataset if dataset else "all"
    return project_root() / output_dir / f"report_{scope}.md"


def write_text(path: Path, content: str) -> None:
    """Write text to a file, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def build_report(
    records: list[dict[str, Any]],
    dataset: str,
    ranking_metric_key: str,
    top_k: int,
    latest_k: int,
    notes_limit: int,
) -> str:
    """Construct the Markdown report body."""
    scope = dataset if dataset else "all datasets"
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    lines = [
        f"# nnU-Net Autoresearch Report: {scope}",
        "",
        f"- Generation time: `{generated_at}`",
        f"- Dataset filter: `{dataset or 'none'}`",
        f"- Experiment count: `{len(records)}`",
        "",
        "## Summary Counts by Status",
        "",
    ]

    counts = status_counts(records)
    if counts:
        lines.append(markdown_table(["Status", "Count"], counts))
    else:
        lines.append("No experiments are registered yet.")

    lines.extend(["", f"## Top Experiments by {ranking_metric_key}", ""])
    top_records = top_by_metric(records, ranking_metric_key, limit=top_k)
    if top_records:
        rows = [
            [
                record.get("experiment_id", ""),
                record.get("title", ""),
                nested_get(record, "results", ranking_metric_key) or "",
                record.get("status", ""),
                record.get("task_type", ""),
            ]
            for record in top_records
        ]
        lines.append(
            markdown_table(
                ["Experiment", "Title", ranking_metric_key, "Status", "Type"],
                rows,
            )
        )
    else:
        lines.append(f"No {ranking_metric_key} metrics are available yet.")

    lines.extend(["", "## Latest Experiments", ""])
    latest = latest_records(records, limit=latest_k)
    if latest:
        rows = [
            [
                record.get("experiment_id", ""),
                record.get("title", ""),
                record.get("created_at", ""),
                record.get("status", ""),
                nested_get(record, "results", ranking_metric_key) or "",
            ]
            for record in latest
        ]
        lines.append(
            markdown_table(
                ["Experiment", "Title", "Created", "Status", ranking_metric_key],
                rows,
            )
        )
    else:
        lines.append("No experiments are available to list.")

    lines.extend(["", "## Brief Notes", ""])
    notes = collect_notes(records, limit=notes_limit)
    if notes:
        lines.extend(notes)
    else:
        lines.append("- No notes or decision rationales have been recorded yet.")

    return "\n".join(lines) + "\n"


def main() -> int:
    """Load registry data and write a Markdown report."""
    args = parse_args()
    global_config = load_global_config()
    reporting_config = global_config.get("reporting", {})
    if not isinstance(reporting_config, dict):
        reporting_config = {}

    ranking_metric_key = metric_result_key(
        reporting_config.get("primary_metric")
        or reporting_config.get("default_primary_metric")
        or global_config.get("default_primary_metric")
        or "dice_mean"
    )
    top_k = int(reporting_config.get("top_k", 5) or 5)
    latest_k = int(reporting_config.get("latest_k", 5) or 5)
    notes_limit = int(reporting_config.get("include_notes_limit", 5) or 5)
    default_output_dir = str(reporting_config.get("default_output_dir", "reports") or "reports")

    jsonl_path = project_root() / "registry" / "experiments.jsonl"
    records = load_jsonl(jsonl_path)

    if args.dataset:
        records = [
            record for record in records if str(record.get("dataset_key", "")) == args.dataset
        ]

    output_path = (
        Path(args.output).expanduser()
        if args.output
        else default_output_path(args.dataset, default_output_dir)
    )
    if not output_path.is_absolute():
        output_path = project_root() / output_path

    report = build_report(
        records,
        args.dataset,
        ranking_metric_key=ranking_metric_key,
        top_k=top_k,
        latest_k=latest_k,
        notes_limit=notes_limit,
    )
    write_text(output_path.resolve(), report)

    print(output_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
