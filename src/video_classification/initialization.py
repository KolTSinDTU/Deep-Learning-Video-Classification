"""Explicit transfer of a trained per-frame encoder to mean late fusion."""

import hashlib
import io
from pathlib import Path

import torch

from .config import PROJECT_ROOT


def initialize_model(model, config: dict, class_to_label: dict) -> dict:
    """Load model weights only; the trainer creates a fresh optimizer afterward.

    This is a warm start, not training resume. Restrict transfer to the shared
    per-frame/late-fusion architecture and the same split regime and labels.
    """
    location = config["training"].get("init_checkpoint")
    if location is None:
        return {"method": "random"}
    if config["model"]["name"] != "late_fusion":
        raise ValueError("init_checkpoint currently supports per_frame -> late_fusion only")
    path = Path(location).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    contents = path.read_bytes()
    checkpoint = torch.load(io.BytesIO(contents), map_location="cpu", weights_only=True)
    required = {"config", "class_to_label", "model_state_dict", "epoch", "smoke_test"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError("init_checkpoint must be a checkpoint saved by this trainer")
    source = checkpoint["config"]
    if source["model"]["name"] != "per_frame":
        raise ValueError("init_checkpoint must come from the per_frame model")
    if checkpoint["smoke_test"]:
        raise ValueError("Cannot initialize an experiment from a smoke-test checkpoint")
    if checkpoint["class_to_label"] != class_to_label:
        raise ValueError("Checkpoint class mapping disagrees with the current training data")
    if source["dataset"]["regime"] != config["dataset"]["regime"]:
        raise ValueError("Checkpoint dataset regime differs; do not transfer original-split "
                         "weights into the corrected-split experiment")
    if source["model"].get("normalization", "none") != config["model"].get("normalization", "none"):
        raise ValueError("Checkpoint and target model must use the same normalization")
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except RuntimeError as error:
        raise ValueError(f"Checkpoint architecture does not match late fusion: {error}") from error
    config["training"]["init_checkpoint"] = str(path)
    return {
        "method": "per_frame_checkpoint",
        "source_checkpoint": str(path),
        "source_sha256": hashlib.sha256(contents).hexdigest(),
        "source_epoch": checkpoint["epoch"],
        "source_model": "per_frame",
        "source_regime": source["dataset"]["regime"],
        "optimizer_state_reused": False,
    }
