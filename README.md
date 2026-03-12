# nnunet-autoresearch

`nnunet-autoresearch` is a local-file experiment operating system for managing nnU-Net v2 work across datasets. Version 1 focuses on disciplined experiment setup, registry updates, and Markdown reporting. It does not modify nnU-Net source code and it does not act as a self-modifying research agent.

The project is intentionally simple:

- Python only
- local files only
- no web UI
- no Docker
- no database server
- reusable across datasets

## First Setup

On a new machine:

1. copy `config/global.example.yaml` to `config/global.yaml`
2. copy `config/datasets/psma5mm.example.yaml` to `config/datasets/psma5mm.yaml`
3. edit the real runtime config files for that machine

Tracked files in Git should remain reusable defaults, code, docs, and templates. Runtime config, registry state, generated reports, and run folders are intentionally local-only so a Linux server can `git pull` without conflicting with machine-specific paths or generated experiment state.

## Folder Structure

```text
nnunet-autoresearch/
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- config/
|   |-- global.yaml
|   |-- experiment_types.yaml
|   `-- datasets/
|       `-- psma5mm.yaml
|-- docs/
|   |-- design_spec.md
|   |-- experiment_policy.md
|   `-- naming_rules.md
|-- registry/
|   |-- experiments.jsonl
|   |-- leaderboard.csv
|   `-- README.md
|-- reports/
|   `-- .gitkeep
|-- runs/
|   `-- .gitkeep
|-- scripts/
|   |-- init_experiment.py
|   |-- update_registry.py
|   |-- build_report.py
|   |-- status.py
|   |-- launch_train.py
|   |-- launch_predict.py
|   |-- launch_postprocess.py
|   `-- collect_metrics.py
`-- templates/
    |-- training.yaml
    |-- inference.yaml
    `-- postprocess.yaml
```

## Phase 1

Phase 1 is fully implemented and provides the repeatable core workflow:

1. initialize an experiment folder under `runs/`
2. update the registry from a run's `meta.yaml`
3. build a Markdown report from the registry

Implemented scripts:

- `scripts/init_experiment.py`
- `scripts/update_registry.py`
- `scripts/build_report.py`

## Phase 2

Phase 2 execution scripts are intentionally stubbed in v1. They are safe placeholders with TODOs and do not launch jobs or mutate experiment state.

Placeholder scripts:

- `scripts/status.py`
- `scripts/launch_train.py`
- `scripts/launch_predict.py`
- `scripts/launch_postprocess.py`
- `scripts/collect_metrics.py`

## Linux Server First-Use

This repository is developed locally but intended to be cloned to a Linux GPU server for real nnU-Net work.

Before first use on the server:

- copy `config/global.example.yaml` to `config/global.yaml`
- copy `config/datasets/psma5mm.example.yaml` to `config/datasets/psma5mm.yaml`
- edit `config/global.yaml`
- edit dataset path config such as `config/datasets/psma5mm.yaml`
- confirm `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results` point to real Linux paths
- review each generated `command.sh`, which is Linux-oriented by design

## Expected Workflow

```text
init experiment -> update registry -> build report
```

PowerShell example:

```powershell
python .\scripts\init_experiment.py `
  --dataset psma5mm `
  --type training `
  --title "Baseline fold0 3d_fullres" `
  --fold 0 `
  --trainer nnUNetTrainer `
  --configuration 3d_fullres `
  --objective "Establish a reproducible baseline for PSMA lesion segmentation." `
  --hypothesis "Default nnUNetTrainer on fold 0 will provide a stable Dice baseline." `
  --change-type baseline `
  --device cuda:0

python .\scripts\update_registry.py --meta .\runs\exp_0001_baseline_fold0_3d_fullres\meta.yaml

python .\scripts\build_report.py --dataset psma5mm --output .\reports\psma5mm_report.md
```

Bash example:

```bash
python scripts/init_experiment.py \
  --dataset psma5mm \
  --type training \
  --title "Baseline fold0 3d_fullres" \
  --fold 0 \
  --trainer nnUNetTrainer \
  --configuration 3d_fullres \
  --objective "Establish a reproducible baseline for PSMA lesion segmentation." \
  --hypothesis "Default nnUNetTrainer on fold 0 will provide a stable Dice baseline." \
  --change-type baseline \
  --device cuda:0

python scripts/update_registry.py --meta runs/exp_0001_baseline_fold0_3d_fullres/meta.yaml

python scripts/build_report.py --dataset psma5mm --output reports/psma5mm_report.md
```

Additional examples:

```bash
python scripts/init_experiment.py \
  --dataset psma5mm \
  --type inference \
  --title "Fold0 validation inference" \
  --fold 0 \
  --parent exp_0001 \
  --objective "Generate validation predictions from the best checkpoint." \
  --hypothesis "Best-checkpoint inference will match validation-time Dice estimates."

python scripts/init_experiment.py \
  --dataset psma5mm \
  --type postprocess \
  --title "Connected component filter 100 voxels" \
  --fold 0 \
  --parent exp_0002 \
  --objective "Reduce false positives with connected-component filtering." \
  --hypothesis "Small-component removal will improve precision without major Dice loss." \
  --change-type ccfilter100
```

## Notes

- Every experiment gets its own folder under `runs/exp_XXXX_*`.
- Every experiment includes `meta.yaml`, `command.sh`, `summary.md`, and `metrics.json`.
- The registry is maintained in both `registry/experiments.jsonl` and `registry/leaderboard.csv`.
- Reports are generated as Markdown files under `reports/`.
- Launch, predict, postprocess, status, and metric collection automation are placeholders in v1 by design.
