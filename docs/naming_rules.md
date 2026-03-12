# Naming Rules

Experiment folders must follow:

```text
exp_XXXX_slug
```

Where:

- `XXXX` is a zero-padded sequential identifier
- `slug` is a lowercase underscore-separated summary of the experiment

Recommended examples:

- `exp_0001_baseline_fold0_3d_fullres`
- `exp_0002_fold0_post_ccfilter100`

Naming guidance:

- use `baseline` for default first-pass training runs
- include fold when relevant
- include postprocessing action for postprocess runs
- keep names short enough to scan in terminal listings
- avoid spaces, punctuation, and dataset-specific noise unless needed
