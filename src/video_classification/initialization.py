"""Explicit transfer of a trained per-frame encoder into video models.

Supported targets:
- late_fusion:  identical parameters, loaded strictly.
- early_fusion: every layer copied; the RGB input kernel is repeated over the
  stacked frames and divided by their number (cross-modality initialization).
- cnn3d:        vgg_bn only; every 2D kernel is inflated along time and
  divided by its temporal size (I3D, Carreira & Zisserman 2017), so a static
  clip initially reproduces the per-frame features.
- dual_stream:  the RGB encoder only; the flow encoder keeps its own weights.
"""

import hashlib
import io
from pathlib import Path

import torch

from .config import PROJECT_ROOT
from .models.backbone import inflate_input_weight

TARGETS = {"late_fusion", "early_fusion", "cnn3d", "dual_stream"}


def _transfer(model, name: str, source_state: dict) -> None:
    if name == "late_fusion":
        model.load_state_dict(source_state, strict=True)
        return
    if name == "dual_stream":
        features = {key[len("features."):]: value for key, value in source_state.items()
                    if key.startswith("features.")}
        model.rgb_features.load_state_dict(features, strict=True)
        return
    target = model.state_dict()
    if set(target) != set(source_state):
        raise RuntimeError(f"Parameter names differ: {sorted(set(target) ^ set(source_state))[:5]}")
    converted = {}
    for key, value in source_state.items():
        shape = target[key].shape
        if key == "features.0.weight":
            # Input convolution: widen from RGB to stacked frames.
            value = inflate_input_weight(value, shape[1])
        if name == "cnn3d" and value.ndim == 4 and len(shape) == 5:
            value = value.unsqueeze(2).repeat(1, 1, shape[2], 1, 1) / shape[2]
        if value.shape != shape:
            raise RuntimeError(f"Cannot map {key}: {tuple(value.shape)} -> {tuple(shape)}")
        converted[key] = value
    model.load_state_dict(converted, strict=True)


def initialize_model(model, config: dict, class_to_label: dict) -> dict:
    """Load model weights only; the trainer creates a fresh optimizer afterward.

    This is a warm start, not training resume. The source must be a per-frame
    checkpoint with the same backbone, split regime, and labels.
    """
    location = config["training"].get("init_checkpoint")
    if location is None:
        return {"method": "random" if not config["model"].get("pretrained") else "imagenet"}
    name = config["model"]["name"]
    if name not in TARGETS:
        raise ValueError(f"init_checkpoint supports per_frame -> {sorted(TARGETS)} only")
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
    backbone = config["model"].get("backbone", "small")
    if source["model"].get("backbone", "small") != backbone:
        raise ValueError("Checkpoint and target model must use the same backbone")
    if name == "cnn3d" and backbone != "vgg_bn":
        raise ValueError("Inflating per-frame weights into cnn3d requires the vgg_bn backbone")
    try:
        _transfer(model, name, checkpoint["model_state_dict"])
    except RuntimeError as error:
        raise ValueError(f"Checkpoint architecture does not match {name}: {error}") from error
    config["training"]["init_checkpoint"] = str(path)
    return {
        "method": "per_frame_checkpoint",
        "transfer": {"late_fusion": "strict copy", "early_fusion": "input-kernel repeat / T",
                     "cnn3d": "I3D temporal inflation", "dual_stream": "RGB encoder only"}[name],
        "source_checkpoint": str(path),
        "source_sha256": hashlib.sha256(contents).hexdigest(),
        "source_epoch": checkpoint["epoch"],
        "source_model": "per_frame",
        "source_regime": source["dataset"]["regime"],
        "optimizer_state_reused": False,
    }
