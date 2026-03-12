"""Launch an nnU-Net v2 training experiment from a run folder."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch a training experiment from an nnU-Net autoresearch run folder."
    )
    parser.add_argument(
        "--exp",
        required=True,
        help="Experiment id such as exp_0003 or a run folder path.",
    )
    parser.add_argument(
        "--tmux",
        action="store_true",
        help="Launch in a detached tmux session.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and write the command without launching it.",
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


def save_yaml(path: Path, payload: dict[str, Any]) -> None:
    """Write a dictionary to YAML with stable formatting."""
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


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


def shell_quote(value: Any) -> str:
    """Quote a value for safe inclusion in a POSIX shell command."""
    text = str(value)
    return "'" + text.replace("'", "'\"'\"'") + "'"


def require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    """Return a required mapping field."""
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"meta.yaml field '{key}' must be a mapping")
    return value


def require_text(payload: dict[str, Any], field_name: str, context: str) -> str:
    """Return a required non-empty string field."""
    value = payload.get(field_name, "")
    text = str(value).strip()
    if not text:
        raise ValueError(f"Missing required {context}: {field_name}")
    return text


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
            + ", ".join(str(path.name) for path in matches)
        )

    raise FileNotFoundError(f"Could not resolve experiment: {exp_arg}")


def extract_nnunet_env(global_config: dict[str, Any]) -> dict[str, str]:
    """Extract required nnU-Net environment paths from global config."""
    env_config = global_config.get("nnunet_env", {})
    if isinstance(env_config, dict):
        raw = str(env_config.get("nnUNet_raw", "")).strip()
        preprocessed = str(env_config.get("nnUNet_preprocessed", "")).strip()
        results = str(env_config.get("nnUNet_results", "")).strip()
        if raw and preprocessed and results:
            return {
                "nnUNet_raw": raw,
                "nnUNet_preprocessed": preprocessed,
                "nnUNet_results": results,
            }

    raw = str(global_config.get("nnUNet_raw", "")).strip()
    preprocessed = str(global_config.get("nnUNet_preprocessed", "")).strip()
    results = str(global_config.get("nnUNet_results", "")).strip()
    if raw and preprocessed and results:
        return {
            "nnUNet_raw": raw,
            "nnUNet_preprocessed": preprocessed,
            "nnUNet_results": results,
        }

    raise ValueError(
        "config/global.yaml must define nnU-Net paths under 'nnunet_env' or at the top level: "
        "nnUNet_raw, nnUNet_preprocessed, nnUNet_results"
    )


def validate_training_context(
    meta: dict[str, Any], global_config: dict[str, Any], dataset_config: dict[str, Any]
) -> dict[str, str]:
    """Validate training metadata and runtime config and return resolved values."""
    if str(meta.get("task_type", "")).strip() != "training":
        raise ValueError("launch_train.py only supports experiments with task_type=training")

    dataset_key = require_text(meta, "dataset_key", "meta.yaml field")
    inputs = require_mapping(meta, "inputs")
    execution = require_mapping(meta, "execution")

    trainer = require_text(inputs, "trainer", "inputs field")
    configuration = require_text(inputs, "configuration", "inputs field")
    fold = require_text(inputs, "fold", "inputs field")
    device = require_text(execution, "device", "execution field")
    dataset_id = require_text(dataset_config, "dataset_id", "dataset config field")

    nnunet_env = extract_nnunet_env(global_config)

    return {
        "dataset_key": dataset_key,
        "dataset_id": dataset_id,
        "trainer": trainer,
        "configuration": configuration,
        "fold": fold,
        "device": device,
        **nnunet_env,
    }


def build_command(resolved: dict[str, str]) -> str:
    """Build the final single-line Linux shell command."""
    return " ".join(
        [
            f"export nnUNet_raw={shell_quote(resolved['nnUNet_raw'])};",
            f"export nnUNet_preprocessed={shell_quote(resolved['nnUNet_preprocessed'])};",
            f"export nnUNet_results={shell_quote(resolved['nnUNet_results'])};",
            "nnUNetv2_train",
            shell_quote(resolved["dataset_id"]),
            shell_quote(resolved["configuration"]),
            shell_quote(resolved["fold"]),
            "-tr",
            shell_quote(resolved["trainer"]),
        ]
    )


def ensure_bash_available() -> str:
    """Ensure a bash executable is available for Linux-oriented command launch."""
    bash_path = shutil.which("bash")
    if not bash_path:
        raise RuntimeError(
            "bash is required to launch training commands because command.sh is Linux-oriented."
        )
    return bash_path


def check_tmux_available() -> str:
    """Ensure tmux is installed."""
    tmux_path = shutil.which("tmux")
    if not tmux_path:
        raise RuntimeError("tmux was requested but is not installed or not on PATH.")
    return tmux_path


def write_command_script(path: Path, command: str) -> None:
    """Write the exact final command to command.sh."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(command + "\n")


def launch_with_tmux(
    run_dir: Path,
    bash_path: str,
    command: str,
    stdout_path: Path,
    stderr_path: Path,
    session_name: str,
) -> None:
    """Launch the training command in a detached tmux session."""
    tmux_command = (
        f"{command} > {shell_quote(stdout_path)} 2> {shell_quote(stderr_path)}"
    )
    subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            session_name,
            bash_path,
            "-lc",
            tmux_command,
        ],
        cwd=run_dir,
        check=True,
    )


def launch_in_foreground(
    run_dir: Path, command: str, stdout_path: Path, stderr_path: Path
) -> int:
    """Launch the training command in the current shell and block until it exits."""
    bash_path = ensure_bash_available()
    with stdout_path.open("w", encoding="utf-8", newline="\n") as stdout_handle:
        with stderr_path.open("w", encoding="utf-8", newline="\n") as stderr_handle:
            completed = subprocess.run(
                [bash_path, "-lc", command],
                cwd=run_dir,
                stdout=stdout_handle,
                stderr=stderr_handle,
                check=False,
            )
    return completed.returncode


def main() -> int:
    """Resolve, validate, and launch an nnU-Net training command."""
    args = parse_args()
    root = project_root()
    run_dir = resolve_experiment_path(args.exp, root)
    meta_path = run_dir / "meta.yaml"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.yaml in run folder: {run_dir}")

    meta = load_yaml(meta_path)
    experiment_id = require_text(meta, "experiment_id", "meta.yaml field")

    if not args.dry_run and str(meta.get("status", "")).strip() == "running":
        raise RuntimeError(
            f"Experiment {experiment_id} is already marked running. Use --dry-run to inspect the command."
        )

    dataset_key = require_text(meta, "dataset_key", "meta.yaml field")
    global_config = load_global_config(root)
    dataset_config = load_dataset_config(root, dataset_key)
    resolved = validate_training_context(meta, global_config, dataset_config)
    command = build_command(resolved)

    command_path = run_dir / "command.sh"
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    write_command_script(command_path, command)

    execution = require_mapping(meta, "execution")
    execution["command"] = command

    mode = "dry-run"
    session_name = ""

    print(f"Run folder: {run_dir}")
    print(f"Command: {command}")
    print(f"stdout log: {stdout_path}")
    print(f"stderr log: {stderr_path}")

    if args.dry_run:
        save_yaml(meta_path, meta)
        print("Mode: dry-run")
    else:
        if args.tmux:
            bash_path = ensure_bash_available()
            check_tmux_available()
            session_name = str(execution.get("tmux_session", "")).strip() or experiment_id
            launch_with_tmux(
                run_dir,
                bash_path,
                command,
                stdout_path,
                stderr_path,
                session_name,
            )
            execution["started_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            execution["tmux_session"] = session_name
            meta["status"] = "running"
            save_yaml(meta_path, meta)
            mode = "tmux"
            print("Mode: tmux")
            print(f"tmux session: {session_name}")
        else:
            ensure_bash_available()
            execution["started_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            meta["status"] = "running"
            save_yaml(meta_path, meta)
            mode = "foreground"
            print("Mode: foreground")
            return_code = launch_in_foreground(run_dir, command, stdout_path, stderr_path)
            print(f"Foreground process exited with code: {return_code}")
            if return_code != 0:
                return return_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
