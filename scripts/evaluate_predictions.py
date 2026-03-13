"""Evaluate predicted segmentation masks against labelsTr using binary Dice."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

import yaml


def project_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate one inference run against labelsTr using binary Dice."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--exp",
        help="Experiment id such as exp_0004 or a run folder path.",
    )
    input_group.add_argument(
        "--prediction-dir",
        help="Direct prediction directory to evaluate without a run folder.",
    )
    parser.add_argument(
        "--dataset-key",
        help="Dataset key for direct prediction-dir mode.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the written evaluation payload as JSON.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-case Dice lines in text mode.",
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


def require_text(payload: dict[str, Any], key: str, context: str) -> str:
    """Return a required non-empty text field."""
    value = str(payload.get(key, "")).strip()
    if not value:
        raise ValueError(f"Missing required {context}: {key}")
    return value


def resolve_nnunet_raw(global_config: dict[str, Any]) -> str:
    """Resolve nnUNet_raw from supported global config locations."""
    env_config = global_config.get("nnunet_env", {})
    if isinstance(env_config, dict):
        value = str(env_config.get("nnUNet_raw", "")).strip()
        if value:
            return value

    value = str(global_config.get("nnUNet_raw", "")).strip()
    if value:
        return value

    raise ValueError(
        "config/global.yaml must define nnUNet_raw under 'nnunet_env' or at the top level"
    )


def ensure_simpleitk() -> Any:
    """Import SimpleITK lazily and fail clearly if it is unavailable."""
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError(
            "SimpleITK is required for evaluate_predictions.py. Install it in the current environment."
        ) from exc
    return sitk


def resolve_paths(
    run_dir: Path, dataset_id: str, global_config: dict[str, Any]
) -> tuple[Path, Path]:
    """Resolve the prediction directory and labelsTr directory."""
    prediction_dir = run_dir / "artifacts" / "predictions"
    return resolve_prediction_and_gt_dirs(prediction_dir, dataset_id, global_config)


def resolve_prediction_and_gt_dirs(
    prediction_dir: Path, dataset_id: str, global_config: dict[str, Any]
) -> tuple[Path, Path]:
    """Resolve the prediction directory and labelsTr directory."""
    if not prediction_dir.exists() or not prediction_dir.is_dir():
        raise FileNotFoundError(f"Prediction directory does not exist: {prediction_dir}")

    ground_truth_dir = Path(resolve_nnunet_raw(global_config)) / dataset_id / "labelsTr"
    if not ground_truth_dir.exists() or not ground_truth_dir.is_dir():
        raise FileNotFoundError(f"Ground-truth labelsTr directory does not exist: {ground_truth_dir}")

    return prediction_dir.resolve(), ground_truth_dir.resolve()


def list_prediction_files(prediction_dir: Path) -> list[Path]:
    """List prediction NIfTI files for evaluation."""
    return sorted(path for path in prediction_dir.glob("*.nii.gz") if path.is_file())


def case_id_from_path(path: Path) -> str:
    """Return a case id from a .nii.gz filename."""
    if path.name.endswith(".nii.gz"):
        return path.name[:-7]
    return path.stem


def match_ground_truth(prediction_path: Path, ground_truth_dir: Path) -> Path | None:
    """Match a prediction file to the exact same basename in labelsTr."""
    candidate = ground_truth_dir / prediction_path.name
    if candidate.exists() and candidate.is_file():
        return candidate.resolve()
    return None


def count_foreground_voxels(image: Any, sitk: Any) -> int:
    """Count foreground voxels in a binary image."""
    stats = sitk.StatisticsImageFilter()
    stats.Execute(image)
    return int(round(stats.GetSum()))


def evaluate_case(prediction_path: Path, ground_truth_path: Path, sitk: Any) -> dict[str, Any]:
    """Compute binary Dice and voxel counts for one matched prediction/label pair."""
    prediction_image = sitk.ReadImage(str(prediction_path))
    ground_truth_image = sitk.ReadImage(str(ground_truth_path))

    if prediction_image.GetSize() != ground_truth_image.GetSize():
        raise ValueError(
            f"Image size mismatch for {prediction_path.name}: "
            f"{prediction_image.GetSize()} vs {ground_truth_image.GetSize()}"
        )

    pred_binary = sitk.Cast(prediction_image > 0, sitk.sitkUInt8)
    gt_binary = sitk.Cast(ground_truth_image > 0, sitk.sitkUInt8)
    intersection = sitk.Multiply(pred_binary, gt_binary)

    pred_voxels = count_foreground_voxels(pred_binary, sitk)
    gt_voxels = count_foreground_voxels(gt_binary, sitk)
    intersection_voxels = count_foreground_voxels(intersection, sitk)

    if pred_voxels == 0 and gt_voxels == 0:
        dice = 1.0
    else:
        dice = (2.0 * intersection_voxels) / float(pred_voxels + gt_voxels)

    return {
        "case_id": case_id_from_path(prediction_path),
        "prediction_file": str(prediction_path),
        "ground_truth_file": str(ground_truth_path),
        "status": "evaluated",
        "dice": dice,
        "pred_voxels": pred_voxels,
        "gt_voxels": gt_voxels,
        "intersection_voxels": intersection_voxels,
    }


def missing_gt_record(prediction_path: Path) -> dict[str, Any]:
    """Build a per-case record for a prediction without matching ground truth."""
    return {
        "case_id": case_id_from_path(prediction_path),
        "prediction_file": str(prediction_path),
        "ground_truth_file": "",
        "status": "missing_gt",
        "dice": None,
        "pred_voxels": None,
        "gt_voxels": None,
        "intersection_voxels": None,
    }


def build_summary(per_case: list[dict[str, Any]]) -> dict[str, Any]:
    """Build evaluation summary statistics."""
    evaluated = [item for item in per_case if item["status"] == "evaluated"]
    missing_gt = [item for item in per_case if item["status"] == "missing_gt"]
    dice_values = [float(item["dice"]) for item in evaluated if item["dice"] is not None]

    if dice_values:
        dice_mean = mean(dice_values)
        dice_median = median(dice_values)
        dice_min = min(dice_values)
        dice_max = max(dice_values)
    else:
        dice_mean = None
        dice_median = None
        dice_min = None
        dice_max = None

    return {
        "prediction_case_count": len(per_case),
        "matched_case_count": len(evaluated),
        "missing_gt_count": len(missing_gt),
        "dice_mean": dice_mean,
        "dice_median": dice_median,
        "dice_min": dice_min,
        "dice_max": dice_max,
    }


def build_payload(
    experiment_id: str,
    run_dir: Path,
    dataset_id: str,
    prediction_dir: Path,
    ground_truth_dir: Path,
    per_case: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the evaluation payload written to evaluation.json."""
    summary = build_summary(per_case)
    notes = [
        "Binary Dice only in v1.",
        "All nonzero voxels are treated as foreground.",
    ]
    if summary["missing_gt_count"]:
        notes.append("Some prediction files were missing matching labelsTr ground truth.")
    if summary["matched_case_count"] == 0:
        notes.append("No matched prediction/ground-truth pairs were available for evaluation.")

    return {
        "experiment_id": experiment_id,
        "run_dir": str(run_dir),
        "dataset_id": dataset_id,
        "prediction_dir": str(prediction_dir),
        "ground_truth_dir": str(ground_truth_dir),
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": summary,
        "per_case": per_case,
        "notes": notes,
    }


def render_text(payload: dict[str, Any], verbose: bool) -> str:
    """Render a concise human-readable evaluation summary."""
    summary = payload["summary"]
    lines = [
        f"Run folder: {payload['run_dir']}",
        f"Prediction directory: {payload['prediction_dir']}",
        f"Ground-truth directory: {payload['ground_truth_dir']}",
        f"Prediction cases: {summary['prediction_case_count']}",
        f"Matched cases: {summary['matched_case_count']}",
        f"Missing ground truth: {summary['missing_gt_count']}",
        f"Dice mean: {summary['dice_mean']}",
        f"Dice median: {summary['dice_median']}",
        f"Dice min: {summary['dice_min']}",
        f"Dice max: {summary['dice_max']}",
    ]

    if verbose and payload["per_case"]:
        lines.append("Per-case Dice:")
        for item in payload["per_case"]:
            if item["status"] == "evaluated":
                lines.append(f"- {item['case_id']}: dice={item['dice']}")
            else:
                lines.append(f"- {item['case_id']}: status={item['status']}")

    return "\n".join(lines)


def resolve_evaluation_context(
    args: argparse.Namespace, root: Path
) -> tuple[str | None, Path | None, Path, str, Path, Path]:
    """Resolve shared evaluation inputs for experiment or direct mode."""
    global_config = load_global_config(root)

    if args.exp:
        run_dir = resolve_experiment_path(args.exp, root)
        meta_path = run_dir / "meta.yaml"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing meta.yaml in run folder: {run_dir}")

        meta = load_yaml(meta_path)
        if str(meta.get("task_type", "")).strip() != "inference":
            raise ValueError(
                "evaluate_predictions.py only supports experiments with task_type=inference"
            )

        experiment_id = require_text(meta, "experiment_id", "meta.yaml field")
        dataset_key = require_text(meta, "dataset_key", "meta.yaml field")
        dataset_config = load_dataset_config(root, dataset_key)
        dataset_id = require_text(dataset_config, "dataset_id", "dataset config field")
        prediction_dir, ground_truth_dir = resolve_paths(run_dir, dataset_id, global_config)
        output_path = run_dir / "evaluation.json"
        return (
            experiment_id,
            run_dir,
            prediction_dir,
            dataset_id,
            ground_truth_dir,
            output_path,
        )

    if not args.dataset_key:
        raise ValueError("--dataset-key is required when using --prediction-dir")

    prediction_dir = Path(args.prediction_dir).expanduser().resolve()
    if not prediction_dir.exists() or not prediction_dir.is_dir():
        raise FileNotFoundError(f"Prediction directory does not exist: {prediction_dir}")

    dataset_config = load_dataset_config(root, args.dataset_key)
    dataset_id = require_text(dataset_config, "dataset_id", "dataset config field")
    prediction_dir, ground_truth_dir = resolve_prediction_and_gt_dirs(
        prediction_dir, dataset_id, global_config
    )
    output_path = prediction_dir / "evaluation.json"
    return None, None, prediction_dir, dataset_id, ground_truth_dir, output_path


def main() -> int:
    """Evaluate one inference run and write evaluation.json."""
    args = parse_args()
    root = project_root()
    (
        experiment_id,
        run_dir,
        prediction_dir,
        dataset_id,
        ground_truth_dir,
        output_path,
    ) = resolve_evaluation_context(args, root)
    prediction_files = list_prediction_files(prediction_dir)
    sitk = ensure_simpleitk()

    per_case: list[dict[str, Any]] = []
    for prediction_path in prediction_files:
        ground_truth_path = match_ground_truth(prediction_path, ground_truth_dir)
        if ground_truth_path is None:
            per_case.append(missing_gt_record(prediction_path))
            continue
        per_case.append(evaluate_case(prediction_path, ground_truth_path, sitk))

    payload = build_payload(
        experiment_id=experiment_id,
        run_dir=run_dir,
        dataset_id=dataset_id,
        prediction_dir=prediction_dir,
        ground_truth_dir=ground_truth_dir,
        per_case=per_case,
    )
    write_json(output_path, payload)

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(render_text(payload, verbose=args.verbose))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
