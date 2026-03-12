# Experiment Policy

The following rules define the operating policy for this repository:

1. never auto-modify nnU-Net source
2. never overwrite previous run outputs
3. never launch on a busy GPU unless explicitly allowed
4. always log a one-line hypothesis before launch
5. always record parent experiment when applicable
6. keep expensive experiments isolated to one major variable at a time

Additional guidance:

- treat `meta.yaml` as the single source of truth for experiment intent and outcomes
- update the registry after meaningful metadata changes
- prefer explicit run titles over ambiguous shorthand
- keep command templates human-reviewable before execution
