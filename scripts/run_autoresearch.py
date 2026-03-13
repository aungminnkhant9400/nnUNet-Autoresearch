"""Run a constrained checkpoint-search autoresearch flow for inference experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

EPSILON = 1e-4


def project_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a constrained checkpoint comparison autoresearch flow."
    )
    parser.add_argument(
        "--baseline-exp",
        required=True,
        help="Baseline inference experiment id such as exp_0004 or a run folder path.",
    )
    parser.add_argument(
        "--candidate-checkpoint",
        required=True,
        help="Candidate checkpoint filename such as checkpoint_final.pth.",
    )
    parser.add_argument(
        "--tmux",
        action="store_true",
        help="Launch candidate inference in a detached tmux session.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create the candidate run and perform prediction dry-run only.",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=20,
        help="Tail length to suggest in follow-up status commands. Default: 20.",
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


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object safely."""
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON file with stable formatting."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


def write_text(path: Path, content: str) -> None:
    """Write a text file with Unix newlines."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


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


def validate_baseline(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the baseline inference run and return its meta and evaluation payloads."""
    meta_path = run_dir / "meta.yaml"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing baseline meta.yaml: {meta_path}")

    meta = load_yaml(meta_path)
    if str(meta.get("task_type", "")).strip() != "inference":
        raise ValueError("run_autoresearch.py only supports baseline experiments with task_type=inference")

    require_text(meta, "experiment_id", "baseline meta.yaml field")
    require_text(meta, "dataset_key", "baseline meta.yaml field")
    inputs = require_mapping(meta, "inputs", "baseline meta.yaml field")
    execution = require_mapping(meta, "execution", "baseline meta.yaml field")
    require_text(inputs, "trainer", "baseline inputs field")
    require_text(inputs, "configuration", "baseline inputs field")
    require_text(inputs, "fold", "baseline inputs field")
    require_text(inputs, "checkpoint", "baseline inputs field")
    require_text(execution, "device", "baseline execution field")

    evaluation_path = run_dir / "evaluation.json"
    if not evaluation_path.exists():
        raise FileNotFoundError(
            f"Baseline evaluation.json is required before checkpoint search: {evaluation_path}"
        )
    evaluation = load_json(evaluation_path)
    summary = evaluation.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"Baseline evaluation.json is missing a valid summary object: {evaluation_path}")

    return meta, evaluation


def build_candidate_metadata(baseline_meta: dict[str, Any], candidate_checkpoint: str) -> dict[str, str]:
    """Build candidate-init parameters derived from the baseline experiment."""
    baseline_id = require_text(baseline_meta, "experiment_id", "baseline meta.yaml field")
    dataset_key = require_text(baseline_meta, "dataset_key", "baseline meta.yaml field")
    inputs = require_mapping(baseline_meta, "inputs", "baseline meta.yaml field")
    execution = require_mapping(baseline_meta, "execution", "baseline meta.yaml field")

    baseline_checkpoint = require_text(inputs, "checkpoint", "baseline inputs field")
    if candidate_checkpoint == baseline_checkpoint:
        raise ValueError("Candidate checkpoint must differ from the baseline checkpoint.")

    fold = require_text(inputs, "fold", "baseline inputs field")
    title = f"fold{fold} inference {candidate_checkpoint} vs {baseline_checkpoint}"

    return {
        "dataset": dataset_key,
        "fold": fold,
        "trainer": require_text(inputs, "trainer", "baseline inputs field"),
        "configuration": require_text(inputs, "configuration", "baseline inputs field"),
        "checkpoint": candidate_checkpoint,
        "parent": baseline_id,
        "title": title,
        "objective": (
            "Compare validation-set inference performance between two checkpoints "
            "for the same fold and configuration."
        ),
        "hypothesis": (
            f"The alternate checkpoint {candidate_checkpoint} may improve validation Dice "
            f"relative to {baseline_checkpoint}."
        ),
        "change_type": "checkpoint_search",
        "change_details": (
            f"Baseline checkpoint: {baseline_checkpoint}. Candidate checkpoint: {candidate_checkpoint}."
        ),
        "device": require_text(execution, "device", "baseline execution field"),
        "baseline_checkpoint": baseline_checkpoint,
        "baseline_experiment_id": baseline_id,
    }


def run_python_script(
    script_name: str,
    script_args: list[str],
    root: Path,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a repository script with the current Python interpreter."""
    command = [sys.executable, str(root / "scripts" / script_name), *script_args]
    return subprocess.run(
        command,
        cwd=root,
        text=True,
        capture_output=capture_output,
        check=False,
    )


def create_candidate_experiment(
    root: Path, candidate_meta: dict[str, str]
) -> Path:
    """Create the candidate inference experiment by reusing init_experiment.py."""
    script_args = [
        "--dataset",
        candidate_meta["dataset"],
        "--type",
        "inference",
        "--title",
        candidate_meta["title"],
        "--fold",
        candidate_meta["fold"],
        "--trainer",
        candidate_meta["trainer"],
        "--configuration",
        candidate_meta["configuration"],
        "--checkpoint",
        candidate_meta["checkpoint"],
        "--parent",
        candidate_meta["parent"],
        "--objective",
        candidate_meta["objective"],
        "--hypothesis",
        candidate_meta["hypothesis"],
        "--change-type",
        candidate_meta["change_type"],
        "--change-details",
        candidate_meta["change_details"],
        "--device",
        candidate_meta["device"],
    ]
    completed = run_python_script("init_experiment.py", script_args, root, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"init_experiment.py failed with exit code {completed.returncode}: "
            f"{completed.stderr or completed.stdout}".strip()
        )

    output_lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not output_lines:
        raise RuntimeError("init_experiment.py did not print the created run path.")

    candidate_path = Path(output_lines[-1]).expanduser()
    if not candidate_path.is_absolute():
        candidate_path = root / candidate_path
    return candidate_path.resolve()


def launch_candidate(root: Path, candidate_run_dir: Path, tmux: bool, dry_run: bool) -> None:
    """Launch the candidate prediction by reusing launch_predict.py."""
    script_args = ["--exp", str(candidate_run_dir)]
    if tmux:
        script_args.append("--tmux")
    if dry_run:
        script_args.append("--dry-run")

    completed = run_python_script("launch_predict.py", script_args, root, capture_output=False)
    if completed.returncode != 0:
        raise RuntimeError(f"launch_predict.py failed with exit code {completed.returncode}")


def evaluate_candidate(root: Path, candidate_run_dir: Path) -> dict[str, Any]:
    """Evaluate the candidate prediction run and return its evaluation payload."""
    completed = run_python_script(
        "evaluate_predictions.py",
        ["--exp", str(candidate_run_dir)],
        root,
        capture_output=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"evaluate_predictions.py failed with exit code {completed.returncode}")

    evaluation_path = candidate_run_dir / "evaluation.json"
    return load_json(evaluation_path)


def numeric_delta(candidate_value: Any, baseline_value: Any) -> float | None:
    """Return candidate minus baseline for numeric values when both are present."""
    if candidate_value is None or baseline_value is None:
        return None
    return float(candidate_value) - float(baseline_value)


def compare_summaries(
    baseline_summary: dict[str, Any], candidate_summary: dict[str, Any]
) -> tuple[dict[str, Any], str, str]:
    """Compare baseline and candidate summaries and return deltas and decision.

    The decision rule stays intentionally simple in v1:
    - prefer the candidate if dice_mean improves by more than EPSILON
    - keep the baseline if dice_mean drops by more than EPSILON
    - otherwise treat the result as inconclusive
    """
    deltas = {
        "dice_mean": numeric_delta(candidate_summary.get("dice_mean"), baseline_summary.get("dice_mean")),
        "dice_median": numeric_delta(candidate_summary.get("dice_median"), baseline_summary.get("dice_median")),
        "dice_min": numeric_delta(candidate_summary.get("dice_min"), baseline_summary.get("dice_min")),
        "dice_max": numeric_delta(candidate_summary.get("dice_max"), baseline_summary.get("dice_max")),
        "matched_case_count": numeric_delta(
            candidate_summary.get("matched_case_count"),
            baseline_summary.get("matched_case_count"),
        ),
        "missing_gt_count": numeric_delta(
            candidate_summary.get("missing_gt_count"),
            baseline_summary.get("missing_gt_count"),
        ),
    }

    delta_mean = deltas["dice_mean"]
    if delta_mean is None:
        decision = "inconclusive"
        rationale = "dice_mean was unavailable in one or both evaluations."
    elif delta_mean > EPSILON:
        decision = "keep_candidate"
        rationale = f"Candidate dice_mean improved by {delta_mean:.6f}."
    elif delta_mean < -EPSILON:
        decision = "keep_baseline"
        rationale = f"Candidate dice_mean dropped by {abs(delta_mean):.6f}."
    else:
        decision = "inconclusive"
        rationale = f"dice_mean difference {delta_mean:.6f} was within epsilon {EPSILON:.1e}."

    return deltas, decision, rationale


def build_comparison_payload(
    baseline_meta: dict[str, Any],
    candidate_meta: dict[str, Any],
    baseline_evaluation: dict[str, Any],
    candidate_evaluation: dict[str, Any],
) -> dict[str, Any]:
    """Build the autoresearch comparison payload."""
    baseline_inputs = require_mapping(baseline_meta, "inputs", "baseline meta.yaml field")
    candidate_inputs = require_mapping(candidate_meta, "inputs", "candidate meta.yaml field")
    baseline_summary = baseline_evaluation["summary"]
    candidate_summary = candidate_evaluation["summary"]
    deltas, decision, rationale = compare_summaries(baseline_summary, candidate_summary)

    return {
        "mode": "checkpoint_search",
        "baseline_experiment_id": require_text(
            baseline_meta, "experiment_id", "baseline meta.yaml field"
        ),
        "candidate_experiment_id": require_text(
            candidate_meta, "experiment_id", "candidate meta.yaml field"
        ),
        "baseline_checkpoint": require_text(
            baseline_inputs, "checkpoint", "baseline inputs field"
        ),
        "candidate_checkpoint": require_text(
            candidate_inputs, "checkpoint", "candidate inputs field"
        ),
        "compared_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "baseline_summary": baseline_summary,
        "candidate_summary": candidate_summary,
        "deltas": deltas,
        "decision": decision,
        "notes": [rationale],
    }


def build_summary_markdown(payload: dict[str, Any]) -> str:
    """Build a human-readable autoresearch comparison summary."""
    baseline_summary = payload["baseline_summary"]
    candidate_summary = payload["candidate_summary"]
    deltas = payload["deltas"]
    rationale = payload["notes"][0] if payload["notes"] else ""
    return (
        "# Autoresearch Checkpoint Comparison\n\n"
        f"- Mode: `{payload['mode']}`\n"
        f"- Baseline Experiment: `{payload['baseline_experiment_id']}`\n"
        f"- Candidate Experiment: `{payload['candidate_experiment_id']}`\n"
        f"- Baseline Checkpoint: `{payload['baseline_checkpoint']}`\n"
        f"- Candidate Checkpoint: `{payload['candidate_checkpoint']}`\n"
        f"- Compared At: `{payload['compared_at']}`\n\n"
        "## Baseline Metrics\n\n"
        f"- dice_mean: `{baseline_summary.get('dice_mean')}`\n"
        f"- dice_median: `{baseline_summary.get('dice_median')}`\n"
        f"- dice_min: `{baseline_summary.get('dice_min')}`\n"
        f"- dice_max: `{baseline_summary.get('dice_max')}`\n"
        f"- matched_case_count: `{baseline_summary.get('matched_case_count')}`\n"
        f"- missing_gt_count: `{baseline_summary.get('missing_gt_count')}`\n\n"
        "## Candidate Metrics\n\n"
        f"- dice_mean: `{candidate_summary.get('dice_mean')}`\n"
        f"- dice_median: `{candidate_summary.get('dice_median')}`\n"
        f"- dice_min: `{candidate_summary.get('dice_min')}`\n"
        f"- dice_max: `{candidate_summary.get('dice_max')}`\n"
        f"- matched_case_count: `{candidate_summary.get('matched_case_count')}`\n"
        f"- missing_gt_count: `{candidate_summary.get('missing_gt_count')}`\n\n"
        "## Deltas\n\n"
        f"- dice_mean: `{deltas.get('dice_mean')}`\n"
        f"- dice_median: `{deltas.get('dice_median')}`\n"
        f"- dice_min: `{deltas.get('dice_min')}`\n"
        f"- dice_max: `{deltas.get('dice_max')}`\n"
        f"- matched_case_count: `{deltas.get('matched_case_count')}`\n"
        f"- missing_gt_count: `{deltas.get('missing_gt_count')}`\n\n"
        "## Decision\n\n"
        f"- Final Decision: `{payload['decision']}`\n"
        f"- Rationale: {rationale}\n"
    )


def write_comparison_artifacts(candidate_run_dir: Path, payload: dict[str, Any]) -> None:
    """Write autoresearch comparison artifacts into the candidate run folder."""
    write_json(candidate_run_dir / "autoresearch_comparison.json", payload)
    write_text(candidate_run_dir / "autoresearch_summary.md", build_summary_markdown(payload))


def main() -> int:
    """Run the checkpoint-search autoresearch flow."""
    args = parse_args()
    root = project_root()
    baseline_run_dir = resolve_experiment_path(args.baseline_exp, root)
    baseline_meta, baseline_evaluation = validate_baseline(baseline_run_dir)
    candidate_config = build_candidate_metadata(baseline_meta, args.candidate_checkpoint)

    print(f"Resolved baseline: {baseline_run_dir}")
    candidate_run_dir = create_candidate_experiment(root, candidate_config)
    candidate_meta_path = candidate_run_dir / "meta.yaml"
    candidate_meta = load_yaml(candidate_meta_path)

    print(f"Created candidate experiment: {candidate_run_dir}")
    launch_candidate(root, candidate_run_dir, tmux=args.tmux, dry_run=args.dry_run)

    if args.dry_run:
        print("Launch mode: dry-run")
        print("Evaluation completed: no")
        print("Comparison completed: no")
        return 0

    if args.tmux:
        candidate_experiment_id = require_text(
            candidate_meta, "experiment_id", "candidate meta.yaml field"
        )
        print("Launch mode: tmux")
        print("Evaluation completed: no")
        print("Comparison completed: no")
        print("Candidate inference was launched in tmux. Evaluate and compare after it finishes.")
        print(
            f"Next step: {sys.executable} {root / 'scripts' / 'status.py'} "
            f"--exp {candidate_experiment_id} --tail {args.tail}"
        )
        print(
            f"Then run: {sys.executable} {root / 'scripts' / 'evaluate_predictions.py'} "
            f"--exp {candidate_experiment_id}"
        )
        return 0

    print("Launch mode: foreground")
    candidate_evaluation = evaluate_candidate(root, candidate_run_dir)
    comparison_payload = build_comparison_payload(
        baseline_meta=baseline_meta,
        candidate_meta=candidate_meta,
        baseline_evaluation=baseline_evaluation,
        candidate_evaluation=candidate_evaluation,
    )
    write_comparison_artifacts(candidate_run_dir, comparison_payload)

    print("Evaluation completed: yes")
    print("Comparison completed: yes")
    print(f"Final decision: {comparison_payload['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
