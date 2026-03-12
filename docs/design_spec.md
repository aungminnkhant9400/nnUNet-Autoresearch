# Design Spec

## Purpose

This repository provides a reusable, file-based operating system for nnU-Net v2 experiments. It manages experiment setup, registry tracking, and Markdown reporting without modifying nnU-Net source code.

## Core Principles

- keep v1 simple and reliable
- store everything in local files
- make experiments reusable across datasets
- prefer explicit metadata over implicit conventions
- separate experiment planning from execution

## Data Model

Each experiment lives in its own run folder:

- `runs/exp_XXXX_*`
- `meta.yaml`
- `command.sh`
- `summary.md`
- `metrics.json`

`meta.yaml` is the authoritative per-run record. Registry files are derived views:

- `registry/experiments.jsonl`: append-style logical registry keyed by `experiment_id`
- `registry/leaderboard.csv`: flattened summary for quick sorting and reporting

## Experiment Classes

The system supports three experiment classes:

1. training
2. inference
3. postprocess

## Phase 1 Scope

Phase 1 implements:

- experiment initialization
- registry synchronization
- Markdown report generation

Phase 1 does not launch jobs, poll cluster state, or parse nnU-Net outputs automatically.

## Phase 2 Scope

Phase 2 is reserved for:

- launch wrappers
- status inspection
- metrics collection
- postprocessing orchestration

Those scripts are intentionally placeholders in v1.
