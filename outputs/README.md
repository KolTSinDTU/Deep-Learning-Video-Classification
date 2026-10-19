# Experiment outputs

The training command creates a timestamped directory for every run, for example
`01_per_frame_seed42/<timestamp>/`. Existing run directories are not reused.
Folders prefixed with `smoke_` contain limited pipeline checks, not experiments.

Training generates:

- `config.json`: effective experiment configuration, including CLI overrides.
- `class_mapping.json`: action names mapped to the course's numeric labels.
- `environment.json`: Python/package versions, device, optimizer, preprocessing.
- `best.pt`: model and optimizer state, configuration, class mapping, epoch,
  and validation metrics. Highest validation video accuracy wins, then lowest
  validation video loss. Smoke-test checkpoints are explicitly marked.
- `history.csv`: epoch, frame training loss/accuracy, video validation
  loss/accuracy, frame/video counts, and epoch duration.
- `summary.json`: completed epochs, best epoch/metrics, checkpoint path,
  elapsed time, and whether the run was a smoke test.

Future final evaluation/reporting should generate:
- `metrics.json`: final video-level metrics, split, seed, and checkpoint identity.
- `predictions.csv`: video identifier, true label, predicted label, class scores.
- `figures/`: learning curves and confusion matrices for the report.

Generated outputs are ignored by Git. The standalone evaluation command remains
to be implemented; training never evaluates the test split.
