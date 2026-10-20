# Experiment outputs

The training command creates a timestamped directory for every run, for example
`01_per_frame_seed42/<timestamp>/`. Existing run directories are not reused.
Folders prefixed with `smoke_` contain limited pipeline checks, not experiments.

Training generates:

- `config.json`: effective experiment configuration, including CLI overrides.
- `class_mapping.json`: action names mapped to the course's numeric labels.
- `environment.json`: Python/package versions, device, optimizer, preprocessing.
  New runs also record whether initialization was random or used a per-frame
  checkpoint, including the source path, SHA-256, and epoch.
- `best.pt`: model and optimizer state, configuration, class mapping, epoch,
  and validation metrics. Highest validation video accuracy wins, then lowest
  validation video loss. Smoke-test checkpoints are explicitly marked.
- `best_loss.pt`: the same checkpoint structure, selected by the lowest
  validation video loss. Saves every new minimum, regardless of the minimum
  decrease required to reset early-stopping patience. Exact loss ties retain
  the earlier checkpoint.
- `history.csv`: epoch, training loss/accuracy, video validation loss/accuracy,
  sample counts, and epoch duration. Per-frame runs use `train_frame_loss`,
  `train_frame_accuracy`, and `train_frames`. Early-fusion, late-fusion, and 3D CNN runs use
  `train_video_loss`, `train_video_accuracy`, and `train_videos`.
  New runs also include `early_stopping_bad_epochs`, the count since the last
  significant validation-loss decrease.
- `summary.json`: completed epochs, best epoch/metrics, checkpoint path,
  elapsed time, and whether the run was a smoke test.
  New runs include `max_epochs`, `stopped_early`, `stop_reason`, and the
  early-stopping state. `best_loss_epoch`, `minimum_val_video_loss`, and
  `best_loss_checkpoint` describe the loss-selected checkpoint; existing
  `best_epoch`, `best_val_video_loss`, and `checkpoint` describe the
  accuracy-selected checkpoint. Stop reasons are `running`, `max_epochs`,
  or `early_stopping`.

Future final evaluation/reporting should generate:
- `metrics.json`: final video-level metrics, split, seed, and checkpoint identity.
- `predictions.csv`: video identifier, true label, predicted label, class scores.
- `figures/`: learning curves and confusion matrices for the report.

Generated outputs are ignored by Git. The standalone evaluation command remains
to be implemented; training never evaluates the test split.

Normalization trials and late-fusion warm starts go under `tuning/`.
The model's saved `normalization` setting controls reconstruction.
Warm starts use a fresh optimizer and include source provenance in `best.pt`;
they are not checkpoint resume. Include the earlier per-frame training budget
when comparing a warm-start run with a from-scratch experiment.
