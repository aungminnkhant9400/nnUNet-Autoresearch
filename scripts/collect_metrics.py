"""Collect conservative validation metrics from common nnU-Net v2 result layouts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

METRIC_KEYS = [
    "dice_mean",
    "dice_median",
    "hd95_mean",
    "precision_mean",
    "recall_mean",
]


def project_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect normalized metrics for one training experiment."
    )
    parser.add_argument(
        "--exp",
        required=True,
        help="Experiment id such as exp_0003 or a run folder path.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the written metrics payload as JSON.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra notes about result-path and file inspection.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file as a dictionary."""
    if not path.exists():
        raise FileNotFoundError(f"Missing YAML file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def load_json(path: Path) -> Any:
    """Load a JSON file safely."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON file with stable formatting."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


def load_global_config(root: Path) -> dict[str, Any]:
    """Load runtime global config with a clear setup error if missing."""
    path = root / "config" / "global.yaml"
    if not path.exists():
        raise FileNotFoundError(
            "Missing runtime config: "
            f"{path}. Copy from {root / 'config' / 'global.example.yaml'} "
            "and edit it for this machine."
        )
    return load_yaml(path)


def load_dataset_config(root: Path, dataset_key: str) -> dict[str, Any]:
    """Load runtime dataset config with a clear setup error if missing."""
    path = root / "config" / "datasets" / f"{dataset_key}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            "Missing dataset runtime config: "
            f"{path}. Copy from {root / 'config' / 'datasets' / f'{dataset_key}.example.yaml'} "
            "and edit it for this machine."
        )
    return load_yaml(path)


def resolve_experiment_path(exp_arg: str, root: Path) -> Path:
    """Resolve an experiment id or run folder path to a run directory."""
    candidate = Path(exp_arg).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    if candidate.is_dir():
        return candidate.resolve()

    experiment_id = exp_arg.strip()
    runs_dir = root / "runs"
    matches = sorted(path for path in runs_dir.glob(f"{experiment_id}_*") if path.is_dir())
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise ValueError(
            f"Multiple run folders matched experiment id {experiment_id}: "
            + ", ".join(path.name for path in matches)
        )
    raise FileNotFoundError(f"Could not resolve experiment: {exp_arg}")


def require_mapping(payload: dict[str, Any], key: str, context: str) -> dict[str, Any]:
    """Return a required mapping field."""
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing required mapping {context}: {key}")
    return value


def require_text(payload: dict[str, Any], key: str, context: str) -> str:
    """Return a required non-empty text field."""
    value = str(payload.get(key, "")).strip()
    if not value:
        raise ValueError(f"Missing required {context}: {key}")
    return value


def resolve_nnunet_results_root(global_config: dict[str, Any]) -> str:
    """Resolve nnUNet_results from supported global config locations."""
    env_config = global_config.get("nnunet_env", {})
    if isinstance(env_config, dict):
        value = str(env_config.get("nnUNet_results", "")).strip()
        if value:
            return value

    value = str(global_config.get("nnUNet_results", "")).strip()
    if value:
        return value

    raise ValueError(
        "config/global.yaml must define nnUNet_results under 'nnunet_env' or at the top level"
    )


def validate_training_context(
    meta: dict[str, Any], global_config: dict[str, Any], dataset_config: dict[str, Any]
) -> dict[str, str]:
    """Validate training metadata and runtime config and return resolved values."""
    if str(meta.get("task_type", "")).strip() != "training":
        raise ValueError("collect_metrics.py only supports experiments with task_type=training")

    dataset_key = require_text(meta, "dataset_key", "meta.yaml field")
    dataset_id = require_text(dataset_config, "dataset_id", "dataset config field")
    inputs = require_mapping(meta, "inputs", "meta.yaml field")

    trainer = require_text(inputs, "trainer", "inputs field")
    configuration = require_text(inputs, "configuration", "inputs field")
    fold = require_text(inputs, "fold", "inputs field")
    nnunet_results = resolve_nnunet_results_root(global_config)

    return {
        "experiment_id": require_text(meta, "experiment_id", "meta.yaml field"),
        "dataset_key": dataset_key,
        "dataset_id": dataset_id,
        "trainer": trainer,
        "configuration": configuration,
        "fold": fold,
        "nnUNet_results": nnunet_results,
        "default_plan": str(dataset_config.get("default_plan", "nnUNetPlans")).strip()
        or "nnUNetPlans",
    }


def resolve_result_dir(context: dict[str, str]) -> tuple[Path | None, list[str]]:
    """Resolve a likely nnU-Net result directory using a small set of common layouts."""
    results_root = Path(context["nnUNet_results"])
    dataset_root = results_root / context["dataset_id"]
    fold_name = f"fold_{context['fold']}"
    trainer = context["trainer"]
    configuration = context["configuration"]
    default_plan = context["default_plan"]

    checked_dirs: list[Path] = []
    for path in (
        dataset_root / f"{trainer}__{default_plan}__{configuration}" / fold_name,
        dataset_root / f"{trainer}__nnUNetPlans__{configuration}" / fold_name,
    ):
        if path not in checked_dirs:
            checked_dirs.append(path)

    for path in checked_dirs:
        if path.exists() and path.is_dir():
            return path.resolve(), [str(item) for item in checked_dirs]

    return None, [str(item) for item in checked_dirs]


def find_candidate_metric_files(result_dir: Path) -> list[Path]:
    """Find a small set of plausible metric JSON files near the fold result directory."""
    candidate_names = {
        "summary.json",
        "metrics.json",
        "validation_summary.json",
        "postprocessing.json",
    }
    candidates: list[Path] = []
    for path in sorted(result_dir.rglob("*.json")):
        try:
            relative_parts = path.relative_to(result_dir).parts
        except ValueError:
            continue
        if len(relative_parts) > 3:
            continue
        lowered_name = path.name.casefold()
        if lowered_name in candidate_names or any(
            marker in lowered_name for marker in ("summary", "metric", "result", "score")
        ):
            candidates.append(path.resolve())
    return candidates


def metric_template() -> dict[str, float | None]:
    """Return an empty normalized metric mapping."""
    return {key: None for key in METRIC_KEYS}


def to_float(value: Any) -> float | None:
    """Convert a value to float if it is clearly numeric."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def extract_from_flat_keys(payload: dict[str, Any]) -> dict[str, float | None]:
    """Extract metrics from direct normalized keys."""
    metrics = metric_template()
    found = False
    for key in METRIC_KEYS:
        value = to_float(payload.get(key))
        if value is not None:
            metrics[key] = value
            found = True
    return metrics if found else {}


def extract_from_foreground_mean(payload: dict[str, Any]) -> dict[str, float | None]:
    """Extract metrics from a common nnU-Net summary structure."""
    metrics = metric_template()
    found = False

    foreground_mean = payload.get("foreground_mean")
    if isinstance(foreground_mean, dict):
        dice = to_float(foreground_mean.get("Dice"))
        hd95 = to_float(foreground_mean.get("HD95"))
        precision = to_float(foreground_mean.get("Precision"))
        recall = to_float(foreground_mean.get("Recall"))
        if dice is not None:
            metrics["dice_mean"] = dice
            found = True
        if hd95 is not None:
            metrics["hd95_mean"] = hd95
            found = True
        if precision is not None:
            metrics["precision_mean"] = precision
            found = True
        if recall is not None:
            metrics["recall_mean"] = recall
            found = True

    foreground_median = payload.get("foreground_median")
    if isinstance(foreground_median, dict):
        dice_median = to_float(foreground_median.get("Dice"))
        if dice_median is not None:
            metrics["dice_median"] = dice_median
            found = True

    return metrics if found else {}


def extract_from_named_metric_maps(payload: dict[str, Any]) -> dict[str, float | None]:
    """Extract metrics from explicit metric-name maps."""
    metrics = metric_template()
    found = False
    key_map = {
        "dice": "dice_mean",
        "hd95": "hd95_mean",
        "precision": "precision_mean",
        "recall": "recall_mean",
    }

    for source_name, target_name in key_map.items():
        metric_payload = payload.get(source_name) or payload.get(source_name.upper())
        if not isinstance(metric_payload, dict):
            continue
        mean_value = to_float(metric_payload.get("mean"))
        if mean_value is not None:
            metrics[target_name] = mean_value
            found = True
        if source_name == "dice":
            median_value = to_float(metric_payload.get("median"))
            if median_value is not None:
                metrics["dice_median"] = median_value
                found = True

    return metrics if found else {}


def extract_metrics(payload: Any) -> dict[str, float | None]:
    """Extract normalized metrics conservatively from a known JSON structure."""
    if not isinstance(payload, dict):
        return {}

    for extractor in (
        extract_from_flat_keys,
        extract_from_foreground_mean,
        extract_from_named_metric_maps,
    ):
        metrics = extractor(payload)
        if metrics:
            return metrics

    return {}


def collect_from_result_dir(result_dir: Path) -> tuple[dict[str, float | None], str, list[str], list[str]]:
    """Collect normalized metrics from candidate JSON files in the result directory."""
    checked_files: list[str] = []
    notes: list[str] = []
    for path in find_candidate_metric_files(result_dir):
        checked_files.append(str(path))
        try:
            payload = load_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            notes.append(f"Skipped unreadable JSON file: {path} ({exc})")
            continue

        metrics = extract_metrics(payload)
        if metrics:
            return metrics, str(path), checked_files, notes

    return metric_template(), "", checked_files, notes


def payload_status(
    result_dir: Path | None,
    source_files_checked: list[str],
    source_file_used: str,
    metrics: dict[str, float | None],
) -> str:
    """Derive the collection status from discovered files and extracted metrics."""
    metric_count = sum(1 for value in metrics.values() if value is not None)
    if metric_count == len(metrics):
        return "found"
    if metric_count > 0:
        return "partial"
    if result_dir is None or not source_files_checked:
        return "missing"
    if not source_file_used:
        return "unsupported_layout"
    return "missing"


def build_payload(
    run_dir: Path,
    context: dict[str, str],
    result_dir: Path | None,
    checked_dirs: list[str],
    source_files_checked: list[str],
    source_file_used: str,
    metrics: dict[str, float | None],
    notes: list[str],
) -> dict[str, Any]:
    """Build the metrics payload written to metrics.json."""
    status = payload_status(result_dir, source_files_checked, source_file_used, metrics)
    final_notes = list(notes)

    if result_dir is None:
        final_notes.append("No supported nnU-Net result directory was found.")
        final_notes.extend(f"Checked result directory: {path}" for path in checked_dirs)
    elif not source_files_checked:
        final_notes.append("No candidate metric JSON files were found in the result directory.")
    elif not source_file_used:
        final_notes.append(
            "Candidate JSON files were found, but none matched the supported metric layouts."
        )

    return {
        "experiment_id": context["experiment_id"],
        "run_dir": str(run_dir),
        "dataset_id": context["dataset_id"],
        "trainer": context["trainer"],
        "configuration": context["configuration"],
        "fold": context["fold"],
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "result_dir_used": str(result_dir) if result_dir else "",
        "source_files_checked": source_files_checked,
        "source_file_used": source_file_used,
        "metrics": metrics,
        "notes": final_notes,
        "status": status,
    }


def render_text(payload: dict[str, Any], verbose: bool) -> str:
    """Render a concise human-readable summary."""
    metrics = payload["metrics"]
    lines = [
        f"Run folder: {payload['run_dir']}",
        f"Result folder: {payload['result_dir_used'] or '(not found)'}",
        f"Source files checked: {len(payload['source_files_checked'])}",
    ]

    if payload["source_files_checked"]:
        lines.extend(f"- {path}" for path in payload["source_files_checked"])

    lines.extend(
        [
            f"Source file used: {payload['source_file_used'] or '(none)'}",
            "Normalized metrics:",
            f"- dice_mean: {metrics['dice_mean']}",
            f"- dice_median: {metrics['dice_median']}",
            f"- hd95_mean: {metrics['hd95_mean']}",
            f"- precision_mean: {metrics['precision_mean']}",
            f"- recall_mean: {metrics['recall_mean']}",
            f"Final status: {payload['status']}",
        ]
    )

    if verbose and payload["notes"]:
        lines.append("Notes:")
        lines.extend(f"- {note}" for note in payload["notes"])

    return "\n".join(lines)


def main() -> int:
    """Collect metrics for one training experiment and write metrics.json."""
    args = parse_args()
    root = project_root()
    run_dir = resolve_experiment_path(args.exp, root)
    meta_path = run_dir / "meta.yaml"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.yaml in run folder: {run_dir}")

    meta = load_yaml(meta_path)
    dataset_key = require_text(meta, "dataset_key", "meta.yaml field")
    global_config = load_global_config(root)
    dataset_config = load_dataset_config(root, dataset_key)
    context = validate_training_context(meta, global_config, dataset_config)
    result_dir, checked_dirs = resolve_result_dir(context)

    if result_dir is None:
        metrics = metric_template()
        source_file_used = ""
        source_files_checked: list[str] = []
        notes: list[str] = []
    else:
        metrics, source_file_used, source_files_checked, notes = collect_from_result_dir(
            result_dir
        )

    payload = build_payload(
        run_dir=run_dir,
        context=context,
        result_dir=result_dir,
        checked_dirs=checked_dirs,
        source_files_checked=source_files_checked,
        source_file_used=source_file_used,
        metrics=metrics,
        notes=notes,
    )

    write_json(run_dir / "metrics.json", payload)

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(render_text(payload, verbose=args.verbose))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
