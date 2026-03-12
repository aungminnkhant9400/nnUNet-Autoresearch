"""Inspect the current operational state of an experiment run folder."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import deque
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
        description="Inspect one nnU-Net autoresearch experiment without modifying it."
    )
    parser.add_argument(
        "--exp",
        required=True,
        help="Experiment id such as exp_0003 or a run folder path.",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=15,
        help="Number of log lines to show from stdout/stderr. Default: 15.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit one JSON object instead of human-readable sections.",
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


def nested_get(payload: dict[str, Any], *keys: str) -> Any:
    """Safely access nested dictionary values."""
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return ""
        current = current.get(key, "")
    return current


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


def iso_timestamp(path: Path) -> str:
    """Return a file's modified time as an ISO 8601 string in UTC."""
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(
        timespec="seconds"
    )


def file_info(path: Path) -> dict[str, Any]:
    """Collect existence, size, and modified time for a file."""
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "size_bytes": 0,
            "modified_at": "",
        }
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "modified_at": iso_timestamp(path),
    }


def read_tail(path: Path, line_count: int) -> list[str]:
    """Read the last N lines from a text file safely."""
    if not path.exists() or path.stat().st_size == 0:
        return []

    buffer: deque[str] = deque(maxlen=max(line_count, 0))
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            buffer.append(line.rstrip("\n"))
    return list(buffer)


def tmux_state(session_name: str) -> str:
    """Check whether a tmux session exists."""
    if not session_name:
        return "missing"

    if not shutil.which("tmux"):
        return "unavailable"

    result = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return "present" if result.returncode == 0 else "missing"


def has_error_text(lines: list[str]) -> bool:
    """Detect obvious error text in recent stderr lines."""
    error_markers = ("error", "exception", "traceback", "runtimeerror", "failed")
    for line in lines:
        lowered = line.casefold()
        if any(marker in lowered for marker in error_markers):
            return True
    return False


def infer_status(
    meta_status: str,
    started_at: str,
    command: str,
    tmux_session_name: str,
    tmux_session_state: str,
    stdout_info: dict[str, Any],
    stderr_info: dict[str, Any],
    stderr_tail: list[str],
) -> str:
    """Infer a simple operational label from metadata, tmux state, and log presence.

    These heuristics stay intentionally small in v1. The goal is quick operator guidance,
    not authoritative job lifecycle management.
    """
    stdout_exists = bool(stdout_info.get("exists"))
    stderr_exists = bool(stderr_info.get("exists"))
    stdout_nonempty = bool(stdout_info.get("size_bytes"))
    stderr_nonempty = bool(stderr_info.get("size_bytes"))

    if meta_status == "running" and tmux_session_state == "present":
        return "running"

    if meta_status == "running" and has_error_text(stderr_tail):
        return "likely_failed"

    if meta_status == "running" and tmux_session_name and tmux_session_state in {"missing", "unavailable"}:
        if stdout_nonempty:
            return "likely_running"
        return "tmux_missing"

    if meta_status == "running" and stdout_nonempty:
        return "likely_running"

    if not started_at and not stdout_exists and not stderr_exists:
        return "not_started"

    if command and not stdout_exists and not stderr_exists:
        return "launched_but_no_logs"

    if stderr_nonempty and has_error_text(stderr_tail):
        return "likely_failed"

    return "unknown"


def render_tail(label: str, path: Path, lines: list[str], info: dict[str, Any]) -> str:
    """Render one log tail section for human-readable output."""
    header = [f"{label}", f"path: {path}"]
    if not info["exists"]:
        header.append("status: missing")
        return "\n".join(header)
    if info["size_bytes"] == 0:
        header.append("status: empty")
        return "\n".join(header)
    header.append("status: present")
    header.append("--")
    header.extend(lines)
    return "\n".join(header)


def render_human(payload: dict[str, Any]) -> str:
    """Render human-readable sections."""
    files = payload["files"]
    lines = [
        "Experiment",
        f"run_dir: {payload['run_dir']}",
        f"experiment_id: {payload['experiment_id']}",
        f"title: {payload['title']}",
        f"task_type: {payload['task_type']}",
        f"dataset_key: {payload['dataset_key']}",
        "",
        "Metadata",
        f"meta_status: {payload['meta_status']}",
        f"device: {payload['device'] or '(missing)'}",
        f"tmux_session: {payload['tmux_session'] or '(none)'}",
        f"started_at: {payload['started_at'] or '(none)'}",
        f"command: {payload['command'] or '(none)'}",
        "",
        "Files",
    ]

    for name, info in files.items():
        lines.append(
            f"{name}: exists={info['exists']} size_bytes={info['size_bytes']} modified_at={info['modified_at'] or '(none)'}"
        )

    lines.extend(
        [
            "",
            "Tmux",
            f"state: {payload['tmux_state']}",
            "",
            "Inferred State",
            payload["inferred_status"],
            "",
            render_tail("stdout tail", Path(files["stdout.log"]["path"]), payload["stdout_tail"], files["stdout.log"]),
            "",
            render_tail("stderr tail", Path(files["stderr.log"]["path"]), payload["stderr_tail"], files["stderr.log"]),
        ]
    )
    return "\n".join(lines)


def main() -> int:
    """Resolve one experiment and report its current state without modifying anything."""
    args = parse_args()
    root = project_root()
    run_dir = resolve_experiment_path(args.exp, root)
    meta_path = run_dir / "meta.yaml"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.yaml in run folder: {run_dir}")

    meta = load_yaml(meta_path)

    files = {
        "meta.yaml": file_info(meta_path),
        "command.sh": file_info(run_dir / "command.sh"),
        "stdout.log": file_info(run_dir / "stdout.log"),
        "stderr.log": file_info(run_dir / "stderr.log"),
        "metrics.json": file_info(run_dir / "metrics.json"),
        "summary.md": file_info(run_dir / "summary.md"),
    }

    stdout_tail = read_tail(run_dir / "stdout.log", args.tail)
    stderr_tail = read_tail(run_dir / "stderr.log", args.tail)

    payload = {
        "run_dir": str(run_dir),
        "experiment_id": str(meta.get("experiment_id", "")),
        "title": str(meta.get("title", "")),
        "task_type": str(meta.get("task_type", "")),
        "dataset_key": str(meta.get("dataset_key", "")),
        "meta_status": str(meta.get("status", "")),
        "device": str(nested_get(meta, "execution", "device")),
        "tmux_session": str(nested_get(meta, "execution", "tmux_session")),
        "tmux_state": tmux_state(str(nested_get(meta, "execution", "tmux_session"))),
        "command": str(nested_get(meta, "execution", "command")),
        "started_at": str(nested_get(meta, "execution", "started_at")),
        "files": files,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }
    payload["inferred_status"] = infer_status(
        meta_status=payload["meta_status"],
        started_at=payload["started_at"],
        command=payload["command"],
        tmux_session_name=payload["tmux_session"],
        tmux_session_state=payload["tmux_state"],
        stdout_info=files["stdout.log"],
        stderr_info=files["stderr.log"],
        stderr_tail=stderr_tail,
    )

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(render_human(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
