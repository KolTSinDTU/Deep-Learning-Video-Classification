"""Load experiment TOML files without importing ML dependencies."""

from pathlib import Path
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAMES = {"per_frame", "late_fusion", "early_fusion", "cnn3d", "dual_stream"}


def early_stopping_settings(training: dict) -> tuple[int, float]:
    """Validate early stopping settings; patience zero disables stopping."""
    patience = training.get("early_stopping_patience", 5)
    min_delta = training.get("early_stopping_min_delta", 0.0001)
    if type(patience) is not int or patience < 0:
        raise ValueError("training.early_stopping_patience must be a nonnegative integer")
    if type(min_delta) not in (int, float) or not 0 <= min_delta < float("inf"):
        raise ValueError("training.early_stopping_min_delta must be finite and nonnegative")
    return patience, float(min_delta)


def load_config(path: str | Path) -> dict:
    """Validate a config and resolve dataset/output paths against the project."""
    with Path(path).open("rb") as stream:
        config = tomllib.load(stream)
    for section in ("experiment", "dataset", "model", "training"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing configuration section: {section}")
    required = {
        "experiment": ("name", "seed", "output_dir"),
        "dataset": ("root", "regime", "num_classes", "num_frames", "image_size"),
        "model": ("name",),
        "training": ("epochs", "batch_size", "learning_rate", "num_workers"),
    }
    for section, keys in required.items():
        for key in keys:
            if key not in config[section]:
                raise ValueError(f"Missing configuration value: {section}.{key}")
    if config["model"]["name"] not in MODEL_NAMES:
        raise ValueError(f"model.name must be one of {sorted(MODEL_NAMES)}")
    if config["model"].get("normalization", "none") not in {"none", "group"}:
        raise ValueError("model.normalization must be none or group")
    if config["dataset"]["regime"] not in {"original", "no_leakage"}:
        raise ValueError("dataset.regime must be original or no_leakage")
    if config["model"]["name"] == "dual_stream" and config["dataset"]["regime"] != "no_leakage":
        raise ValueError("The dual-stream experiment uses the corrected dataset")
    for section, keys in {
        "dataset": ("num_classes", "num_frames", "image_size"),
        "training": ("epochs", "batch_size"),
    }.items():
        for key in keys:
            value = config[section][key]
            if type(value) is not int or value <= 0:
                raise ValueError(f"{section}.{key} must be a positive integer")
    for section, key in (("experiment", "seed"), ("training", "num_workers")):
        value = config[section][key]
        if type(value) is not int or value < 0:
            raise ValueError(f"{section}.{key} must be a nonnegative integer")
    rate = config["training"]["learning_rate"]
    if type(rate) not in (int, float) or not 0 < rate < float("inf"):
        raise ValueError("training.learning_rate must be finite and positive")
    validation_batch = config["training"].get("validation_batch_size", 1)
    if type(validation_batch) is not int or validation_batch <= 0:
        raise ValueError("training.validation_batch_size must be a positive integer")
    if config["training"].get("device", "auto") not in {"auto", "cpu", "cuda"}:
        raise ValueError("training.device must be auto, cpu, or cuda")
    early_stopping_settings(config["training"])
    name = config["experiment"]["name"]
    if not isinstance(name, str) or not name.strip():
        raise ValueError("experiment.name must be a nonempty string")
    path_settings = [("dataset", "root"), ("experiment", "output_dir")]
    if "init_checkpoint" in config["training"]:
        path_settings.append(("training", "init_checkpoint"))
    for section, key in path_settings:
        raw = config[section][key]
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"{section}.{key} must be a nonempty path")
        location = Path(raw).expanduser()
        if not location.is_absolute():
            location = PROJECT_ROOT / location
        config[section][key] = str(location.resolve())
    return config
