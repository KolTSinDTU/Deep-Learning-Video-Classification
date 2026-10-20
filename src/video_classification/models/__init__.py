"""Single-stream model factory; dual stream remains to be implemented."""

from ..config import MODEL_NAMES


def build_model(config: dict):
    """Construct the selected model with dataset.num_classes output logits."""
    name = config["model"]["name"]
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {name}")
    model_options = {
        "num_classes": config["dataset"]["num_classes"],
        "normalization": config["model"].get("normalization", "none"),
    }
    if name == "per_frame":
        from .per_frame import PerFrameCNN

        return PerFrameCNN(**model_options)
    if name == "late_fusion":
        from .late_fusion import LateFusionCNN

        return LateFusionCNN(**model_options)
    if name == "early_fusion":
        from .early_fusion import EarlyFusionCNN

        return EarlyFusionCNN(
            **model_options,
            num_frames=config["dataset"]["num_frames"],
        )
    if name == "cnn3d":
        from .cnn3d import CNN3D

        return CNN3D(**model_options)
    raise NotImplementedError(f"Implement {name} in models/{name}.py and register it here")
