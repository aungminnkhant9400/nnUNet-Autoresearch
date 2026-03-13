"""Apply simple binary connected-component postprocessing to prediction masks."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Apply simple connected-component postprocessing to .nii.gz masks."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing .nii.gz masks.")
    parser.add_argument("--output-dir", required=True, help="Directory to write processed masks.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["none", "largest_component", "min_component_size", "largest_k_components"],
        help="Postprocessing mode.",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=None,
        help="Minimum voxel count for min_component_size mode.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of largest components to keep for largest_k_components mode.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the written summary payload as JSON.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print one line per processed case in text mode.",
    )
    return parser.parse_args()


def ensure_runtime_dependencies() -> tuple[Any, Any]:
    """Import numpy and SimpleITK lazily with clear setup errors."""
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "numpy is required for apply_postprocess.py. Install it in the current environment."
        ) from exc

    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError(
            "SimpleITK is required for apply_postprocess.py. Install it in the current environment."
        ) from exc

    return np, sitk


def resolve_directory(path_text: str) -> Path:
    """Resolve a directory path without requiring it to exist already."""
    return Path(path_text).expanduser().resolve()


def validate_args(args: argparse.Namespace, input_dir: Path, output_dir: Path) -> None:
    """Validate CLI arguments and directory relationships."""
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if input_dir == output_dir:
        raise ValueError("output-dir must be different from input-dir")

    if args.mode == "min_component_size":
        if args.min_size is None or args.min_size <= 0:
            raise ValueError("--min-size must be provided and > 0 for min_component_size mode")

    if args.mode == "largest_k_components":
        if args.top_k is None or args.top_k <= 0:
            raise ValueError("--top-k must be provided and > 0 for largest_k_components mode")


def list_prediction_files(input_dir: Path) -> list[Path]:
    """List input .nii.gz prediction files."""
    return sorted(path for path in input_dir.glob("*.nii.gz") if path.is_file())


def case_id_from_path(path: Path) -> str:
    """Return a case id from a .nii.gz filename."""
    if path.name.endswith(".nii.gz"):
        return path.name[:-7]
    return path.stem


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON file with stable formatting."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


def binary_image_from_input(image: Any, np: Any, sitk: Any) -> tuple[Any, Any]:
    """Convert an input image to a binary numpy array and SimpleITK image."""
    array = sitk.GetArrayFromImage(image)
    binary_array = (array > 0).astype(np.uint8)
    binary_image = sitk.GetImageFromArray(binary_array)
    binary_image.CopyInformation(image)
    return binary_array, binary_image


def component_sizes(binary_image: Any, sitk: Any) -> tuple[Any, dict[int, int]]:
    """Return connected-component labels and voxel counts."""
    connected = sitk.ConnectedComponent(binary_image)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(connected)
    sizes = {
        int(label): int(stats.GetNumberOfPixels(label))
        for label in stats.GetLabels()
    }
    return connected, sizes


def filtered_binary_array(
    binary_image: Any,
    mode: str,
    min_size: int | None,
    top_k: int | None,
    np: Any,
    sitk: Any,
) -> tuple[Any, dict[int, int], dict[int, int]]:
    """Apply the requested connected-component mode and return the filtered binary array."""
    connected_before, sizes_before = component_sizes(binary_image, sitk)

    if mode == "none" or not sizes_before:
        before_array = sitk.GetArrayFromImage(binary_image)
        return before_array.astype(np.uint8), sizes_before, sizes_before

    if mode == "largest_component":
        keep_labels = {max(sizes_before, key=sizes_before.get)}
    elif mode == "min_component_size":
        keep_labels = {label for label, size in sizes_before.items() if size >= int(min_size or 0)}
    elif mode == "largest_k_components":
        keep_count = int(top_k or 0)
        sorted_labels = sorted(
            sizes_before.items(),
            key=lambda item: (-item[1], item[0]),
        )
        keep_labels = {label for label, _ in sorted_labels[:keep_count]}
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    connected_array = sitk.GetArrayFromImage(connected_before)
    if keep_labels:
        filtered_array = np.isin(connected_array, list(keep_labels)).astype(np.uint8)
    else:
        filtered_array = np.zeros_like(connected_array, dtype=np.uint8)

    filtered_image = sitk.GetImageFromArray(filtered_array)
    filtered_image.CopyInformation(binary_image)
    _, sizes_after = component_sizes(filtered_image, sitk)
    return filtered_array, sizes_before, sizes_after


def write_mask(
    output_path: Path,
    source_image: Any,
    binary_array: Any,
    sitk: Any,
) -> None:
    """Write a processed binary mask while preserving metadata."""
    output_image = sitk.GetImageFromArray(binary_array.astype("uint8"))
    output_image.CopyInformation(source_image)
    sitk.WriteImage(sitk.Cast(output_image, sitk.sitkUInt8), str(output_path))


def process_case(
    input_path: Path,
    output_dir: Path,
    mode: str,
    min_size: int | None,
    top_k: int | None,
    np: Any,
    sitk: Any,
) -> dict[str, Any]:
    """Process one mask and return per-case statistics."""
    source_image = sitk.ReadImage(str(input_path))
    binary_array, binary_image = binary_image_from_input(source_image, np, sitk)
    filtered_array, sizes_before, sizes_after = filtered_binary_array(
        binary_image=binary_image,
        mode=mode,
        min_size=min_size,
        top_k=top_k,
        np=np,
        sitk=sitk,
    )

    foreground_before = int(binary_array.sum())
    foreground_after = int(filtered_array.sum())
    output_path = output_dir / input_path.name
    write_mask(output_path, source_image, filtered_array, sitk)

    return {
        "case_id": case_id_from_path(input_path),
        "input_file": str(input_path),
        "output_file": str(output_path),
        "mode": mode,
        "foreground_voxels_before": foreground_before,
        "foreground_voxels_after": foreground_after,
        "removed_voxels": foreground_before - foreground_after,
        "component_count_before": len(sizes_before),
        "component_count_after": len(sizes_after),
        "status": "processed",
    }


def build_summary(per_case: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    """Build summary statistics across processed cases."""
    case_count = len(per_case)
    total_before = sum(int(item["foreground_voxels_before"]) for item in per_case)
    total_after = sum(int(item["foreground_voxels_after"]) for item in per_case)
    total_removed = sum(int(item["removed_voxels"]) for item in per_case)

    if case_count:
        mean_before = total_before / case_count
        mean_after = total_after / case_count
    else:
        mean_before = 0.0
        mean_after = 0.0

    return {
        "case_count": case_count,
        "mode": mode,
        "total_foreground_voxels_before": total_before,
        "total_foreground_voxels_after": total_after,
        "total_removed_voxels": total_removed,
        "mean_foreground_voxels_before": mean_before,
        "mean_foreground_voxels_after": mean_after,
    }


def build_payload(
    input_dir: Path,
    output_dir: Path,
    mode: str,
    min_size: int | None,
    top_k: int | None,
    per_case: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the postprocessing summary payload."""
    notes = [
        "v1 operates only on .nii.gz prediction masks.",
        "All nonzero voxels are treated as foreground.",
        "Connected components use SimpleITK default behavior.",
    ]
    if mode == "none":
        notes.append("Mode none writes binary masks unchanged after foreground binarization.")

    return {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "mode": mode,
        "parameters": {
            "min_size": min_size,
            "top_k": top_k,
        },
        "processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": build_summary(per_case, mode),
        "per_case": per_case,
        "notes": notes,
    }


def render_text(payload: dict[str, Any], verbose: bool) -> str:
    """Render a concise human-readable summary."""
    summary = payload["summary"]
    lines = [
        f"Input directory: {payload['input_dir']}",
        f"Output directory: {payload['output_dir']}",
        f"Mode: {payload['mode']}",
        f"Processed cases: {summary['case_count']}",
        f"Total voxels before: {summary['total_foreground_voxels_before']}",
        f"Total voxels after: {summary['total_foreground_voxels_after']}",
        f"Total removed voxels: {summary['total_removed_voxels']}",
    ]

    if verbose:
        lines.append("Per-case:")
        for item in payload["per_case"]:
            lines.append(
                f"- {item['case_id']}: before={item['foreground_voxels_before']} "
                f"after={item['foreground_voxels_after']} removed={item['removed_voxels']}"
            )

    return "\n".join(lines)


def main() -> int:
    """Run binary connected-component postprocessing over a prediction directory."""
    args = parse_args()
    np, sitk = ensure_runtime_dependencies()

    input_dir = resolve_directory(args.input_dir)
    output_dir = resolve_directory(args.output_dir)
    validate_args(args, input_dir, output_dir)

    input_files = list_prediction_files(input_dir)
    if not input_files:
        raise FileNotFoundError(f"No .nii.gz files were found in input directory: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    per_case = [
        process_case(
            input_path=path,
            output_dir=output_dir,
            mode=args.mode,
            min_size=args.min_size,
            top_k=args.top_k,
            np=np,
            sitk=sitk,
        )
        for path in input_files
    ]
    payload = build_payload(
        input_dir=input_dir,
        output_dir=output_dir,
        mode=args.mode,
        min_size=args.min_size,
        top_k=args.top_k,
        per_case=per_case,
    )
    write_json(output_dir / "postprocess_summary.json", payload)

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(render_text(payload, verbose=args.verbose))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
