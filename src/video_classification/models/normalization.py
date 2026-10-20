"""Optional feature normalization without changing the baseline layer names."""

from collections import OrderedDict

from torch import nn


def normalize_convolutions(features: nn.Sequential, normalization: str) -> nn.Sequential:
    """Insert eight-group normalization between each convolution and ReLU.

    GroupNorm computes statistics separately for each sample, including with
    batches of one. Constructing the existing convolutions first preserves
    their seeded initialization; the added affine parameters start at 1 and 0.
    The default returns the original layers, keeping old checkpoints loadable.
    """
    if normalization == "none":
        return features
    if normalization != "group":
        raise ValueError("model.normalization must be none or group")

    layers = OrderedDict()
    for name, layer in features.named_children():
        layers[name] = layer
        if isinstance(layer, (nn.Conv2d, nn.Conv3d)):
            layers[f"norm_after_{name}"] = nn.GroupNorm(8, layer.out_channels)
    return nn.Sequential(layers)
