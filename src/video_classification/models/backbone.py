"""Shared 2D/3D convolutional backbones used by every video model.

All models build on the same backbone and differ only in where time is fused,
so architecture comparisons are controlled. Each backbone is an nn.Sequential
ending in global average pooling; index 0 is always the input convolution,
which lets fusion models widen it for stacked frames or flow.

- small:    the original 3-conv baseline (no BN); kept for old checkpoints.
- vgg_bn:   4 stages of two conv-BN-ReLU layers, widths 64/128/256/256.
- resnet18: torchvision ResNet-18, optionally ImageNet-pretrained (2D only).
"""

import torch
from torch import nn

from .normalization import normalize_convolutions

BACKBONES = {"small", "vgg_bn", "resnet18"}
VGG_WIDTHS = (64, 128, 256, 256)
# Time is kept for two stages, then halved twice: T=10 -> 10, 10, 5, 2.
VGG_3D_POOLS = ((1, 2, 2), (1, 2, 2), (2, 2, 2), (2, 2, 2))


def _small(dimensions: int, normalization: str) -> tuple[nn.Sequential, int]:
    conv = nn.Conv2d if dimensions == 2 else nn.Conv3d
    pool = (lambda: nn.MaxPool2d(2)) if dimensions == 2 else (lambda: nn.MaxPool3d((1, 2, 2)))
    gap = nn.AdaptiveAvgPool2d(1) if dimensions == 2 else nn.AdaptiveAvgPool3d(1)
    features = nn.Sequential(
        conv(3, 32, kernel_size=3, padding=1), nn.ReLU(), pool(),
        conv(32, 64, kernel_size=3, padding=1), nn.ReLU(), pool(),
        conv(64, 128, kernel_size=3, padding=1), nn.ReLU(), pool(),
        gap,
    )
    return normalize_convolutions(features, normalization), 128


def _vgg_bn(dimensions: int) -> tuple[nn.Sequential, int]:
    conv, norm = (nn.Conv2d, nn.BatchNorm2d) if dimensions == 2 else (nn.Conv3d, nn.BatchNorm3d)
    layers, channels = [], 3
    for stage, width in enumerate(VGG_WIDTHS):
        for _ in range(2):
            layers += [conv(channels, width, kernel_size=3, padding=1, bias=False),
                       norm(width), nn.ReLU(inplace=True)]
            channels = width
        layers.append(nn.MaxPool2d(2) if dimensions == 2 else nn.MaxPool3d(VGG_3D_POOLS[stage]))
    layers.append(nn.AdaptiveAvgPool2d(1) if dimensions == 2 else nn.AdaptiveAvgPool3d(1))
    return nn.Sequential(*layers), channels


def _resnet18(pretrained: bool) -> tuple[nn.Sequential, int]:
    from torchvision.models import ResNet18_Weights, resnet18

    # Pretrained weights download to the torch hub cache on first use; on the
    # cluster, trigger the download once from a login node with internet access.
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    features = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                             net.layer1, net.layer2, net.layer3, net.layer4, net.avgpool)
    return features, 512


def build_backbone(name: str = "small", *, dimensions: int = 2, normalization: str = "none",
                   pretrained: bool = False) -> tuple[nn.Sequential, int]:
    """Return (features, feature_dim) mapping [N, 3, (T,) H, W] to [N, D, 1, 1(, 1)]."""
    if name not in BACKBONES:
        raise ValueError(f"model.backbone must be one of {sorted(BACKBONES)}")
    if name != "small" and normalization != "none":
        raise ValueError("model.normalization applies to the small backbone only; "
                         f"{name} already uses batch normalization")
    if pretrained and name != "resnet18":
        raise ValueError("model.pretrained is only available for the resnet18 backbone")
    if name == "small":
        return _small(dimensions, normalization)
    if name == "vgg_bn":
        return _vgg_bn(dimensions)
    if dimensions != 2:
        raise ValueError("The resnet18 backbone is 2D only; use small or vgg_bn for cnn3d")
    return _resnet18(pretrained)


def widen_input_conv(features: nn.Sequential, in_channels: int, *, inherit: bool) -> None:
    """Replace features[0] with an in_channels convolution, in place.

    With inherit=True the new kernel is derived from the old RGB kernel
    (cross-modality initialization, Wang et al. 2016): stacked RGB frames get
    the RGB kernel repeated and divided by the number of frames, so a static
    clip produces the original response; other inputs (e.g. stacked flow) get
    the channel-mean kernel scaled to keep the same total weight.
    """
    old = features[0]
    new = type(old)(in_channels, old.out_channels, kernel_size=old.kernel_size,
                    stride=old.stride, padding=old.padding, bias=old.bias is not None)
    if inherit:
        with torch.no_grad():
            new.weight.copy_(inflate_input_weight(old.weight, in_channels))
            if old.bias is not None:
                new.bias.copy_(old.bias)
    features[0] = new


def inflate_input_weight(weight: torch.Tensor, in_channels: int) -> torch.Tensor:
    """Map an RGB input kernel [O, 3, ...] to [O, in_channels, ...]."""
    if weight.shape[1] != 3:
        raise ValueError("Expected an RGB input kernel")
    if in_channels % 3 == 0:
        copies = in_channels // 3
        return weight.repeat(1, copies, *([1] * (weight.ndim - 2))) / copies
    mean = weight.mean(dim=1, keepdim=True)
    return mean.expand(-1, in_channels, *weight.shape[2:]) * (3 / in_channels)
