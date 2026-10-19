"""Video-level evaluation implementation starting point."""


def evaluate(config: dict, checkpoint: str, split: str = "val"):
    """Evaluate a saved checkpoint; reserve split='test' for final reporting.

    TODO: restore model/class mapping, use eval mode and inference mode,
    aggregate frame predictions when appropriate, and export video-level
    accuracy, macro F1, per-class metrics, confusion matrix, and predictions.
    Record the checkpoint and split with every result.
    """
    if split not in {"val", "test"}:
        raise ValueError("split must be val or test")
    raise NotImplementedError("Implement checkpoint loading and video-level evaluation")
