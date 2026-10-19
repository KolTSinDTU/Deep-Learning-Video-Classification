"""Model factory interface; architectures remain to be implemented."""

from ..config import MODEL_NAMES


def build_model(config: dict):
    """Construct the selected model with dataset.num_classes output logits."""
    name = config["model"]["name"]
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {name}")
    raise NotImplementedError(f"Implement {name} in models/{name}.py and register it here")
