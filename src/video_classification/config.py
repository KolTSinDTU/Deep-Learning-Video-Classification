"""Load experiment TOML files without importing ML dependencies."""

from pathlib import Path
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAMES = {"per_frame", "late_fusion", "early_fusion", "cnn3d", "dual_stream", "flow_stream"}
BACKBONES = {"small", "vgg_bn", "resnet18"}
OPTIMIZERS = {"adam", "adamw", "sgd"}
SCHEDULERS = {"none", "cosine"}


def validate_dual_stream(data: dict) -> None:
    """Require an explicit flow representation and the corrected split regime."""
    if data.get("regime") != "no_leakage":
        raise ValueError("The dual-stream experiment uses the corrected dataset")
    if type(data.get("num_frames")) is not int or data["num_frames"] < 2:
        raise ValueError("Dual stream requires dataset.num_frames of at least 2")
    if data.get("flow_format") not in {"rgb_png", "uv_npy"}:
        raise ValueError("dataset.flow_format must explicitly be rgb_png or uv_npy")
    if data["flow_format"] == "uv_npy":
        scale = data.get("flow_scale")
        if type(scale) not in (int, float) or not 0 < scale < float("inf"):
            raise ValueError("dataset.flow_scale must be finite and positive for uv_npy")


def validate_image_size(value: int | str) -> None:
    """Allow unchanged source frames or an explicit square resize."""
    if value == "native":
        return
    if type(value) is not int or value < 8:
        raise ValueError('dataset.image_size must be "native" or an integer of at least 8')


def validate_recipe(config: dict) -> None:
    """Check optional backbone, augmentation, and optimizer settings.

    Every setting defaults to the original baseline recipe, so older configs
    and checkpoints keep their meaning.
    """
    model, data, training = config["model"], config["dataset"], config["training"]
    if model.get("backbone", "small") not in BACKBONES:
        raise ValueError(f"model.backbone must be one of {sorted(BACKBONES)}")
    if type(model.get("pretrained", False)) is not bool:
        raise ValueError("model.pretrained must be true or false")
    dropout = model.get("dropout", 0.0)
    if type(dropout) not in (int, float) or not 0 <= dropout < 1:
        raise ValueError("model.dropout must be in [0, 1)")
    if model.get("temporal_pool", "mean") not in {"mean", "max"}:
        raise ValueError("model.temporal_pool must be mean or max")
    if model.get("flow_input", "mean") not in {"mean", "stack"}:
        raise ValueError("model.flow_input must be mean or stack")
    if type(data.get("augment", False)) is not bool:
        raise ValueError("dataset.augment must be true or false")
    if data.get("augment", False) and data["image_size"] == "native":
        raise ValueError("dataset.augment requires an integer dataset.image_size")
    if data.get("input_normalization", "none") not in {"none", "standardize"}:
        raise ValueError("dataset.input_normalization must be none or standardize")
    if training.get("optimizer", "adam") not in OPTIMIZERS:
        raise ValueError(f"training.optimizer must be one of {sorted(OPTIMIZERS)}")
    if training.get("scheduler", "none") not in SCHEDULERS:
        raise ValueError(f"training.scheduler must be one of {sorted(SCHEDULERS)}")
    decay = training.get("weight_decay", 0.0)
    if type(decay) not in (int, float) or not 0 <= decay < float("inf"):
        raise ValueError("training.weight_decay must be finite and nonnegative")
    warmup = training.get("warmup_epochs", 0)
    if type(warmup) is not int or not 0 <= warmup < training["epochs"]:
        raise ValueError("training.warmup_epochs must be an integer in [0, epochs)")
    for key in ("save_optimizer_state", "save_best_loss_checkpoint"):
        if type(training.get(key, True)) is not bool:
            raise ValueError(f"training.{key} must be true or false")
    if type(training.get("label_smoothing", 0.0)) not in (int, float) or not (
            0 <= training.get("label_smoothing", 0.0) < 1):
        raise ValueError("training.label_smoothing must be in [0, 1)")


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
    if config["model"]["name"] in {"dual_stream", "flow_stream"}:
        validate_dual_stream(config["dataset"])
    validate_recipe(config)
    for section, keys in {
        "dataset": ("num_classes", "num_frames"),
        "training": ("epochs", "batch_size"),
    }.items():
        for key in keys:
            value = config[section][key]
            if type(value) is not int or value <= 0:
                raise ValueError(f"{section}.{key} must be a positive integer")
    validate_image_size(config["dataset"]["image_size"])
    if config["dataset"].get("frame_source", "provided") not in {"provided", "native_video"}:
        raise ValueError("dataset.frame_source must be provided or native_video")
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
    if "frame_cache_root" in config["dataset"]:
        path_settings.append(("dataset", "frame_cache_root"))
    if "flow_manifest_root" in config["dataset"]:
        path_settings.append(("dataset", "flow_manifest_root"))
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
