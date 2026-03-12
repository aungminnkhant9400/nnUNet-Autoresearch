# Registry

The registry stores durable experiment tracking artifacts.

Files:

- `experiments.jsonl`: authoritative experiment records keyed by `experiment_id`
- `leaderboard.csv`: flattened view for quick sorting and reporting

Usage notes:

- update the registry after creating an experiment and after any meaningful result update
- `experiments.jsonl` preserves the nested metadata structure from `meta.yaml`
- `leaderboard.csv` is regenerated from the JSONL registry to avoid duplicate rows
