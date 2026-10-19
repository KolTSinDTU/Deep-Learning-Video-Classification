# Experiment outputs

Use a separate directory per run, for example `01_per_frame_seed42/`:

- `config.toml`: exact experiment configuration.
- `environment.txt`: Python/package versions and device information.
- `best.pt`: checkpoint chosen by validation performance.
- `history.csv`: epoch, training loss, validation loss, validation accuracy.
- `metrics.json`: final video-level metrics, split, seed, and checkpoint identity.
- `predictions.csv`: video identifier, true label, predicted label, class scores.
- `figures/`: learning curves and confusion matrices for the report.

Generated outputs are ignored by Git. These are recommended output conventions;
the training and evaluation scaffolds do not generate them yet.
