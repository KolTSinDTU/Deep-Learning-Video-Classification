"""Training implementation starting point."""


def train(config: dict):
    """Train using training data and select checkpoints using validation only.

    TODO: seed Python/NumPy/PyTorch and workers; build loaders/model; choose
    device/optimizer/loss; implement train/eval mode changes and epoch loops;
    save the best validation checkpoint, config, class mapping, and history.
    Protect existing run outputs from accidental overwrite. Do not evaluate
    the test split during model selection. See outputs/README.md.
    """
    raise NotImplementedError("Implement the training loop after the data adapter and model")
