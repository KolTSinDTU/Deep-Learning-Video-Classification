"""RGB/flow two-stream models (Simonyan & Zisserman, 2014).

FlowStreamCNN is the temporal stream on its own: all flow fields of a clip
are stacked along channels, so the first convolution sees motion trajectories
rather than isolated displacements. As in the original paper, it is trained
separately from the RGB stream and the two are combined by weighted averaging
of softmax scores (see video_classification.fuse).

DualStreamCNN is the jointly trained alternative with feature-level fusion.
"""

import torch
from torch import Tensor, nn

from .backbone import widen_input_conv
from .early_fusion import EarlyFusionCNN
from .per_frame import PerFrameCNN

FLOW_INPUTS = {"mean", "stack"}


class FlowStreamCNN(EarlyFusionCNN):
    """Classify a video from its stacked flow fields [B, C, S, H, W].

    C is 3 for colour-coded flow images or 2 for horizontal/vertical vectors;
    S is the number of flow fields (normally 9 for 10 frames).
    """

    def __init__(self, num_classes: int = 10, *, flow_channels: int = 3,
                 num_flow_frames: int = 9, **options):
        if type(flow_channels) is not int or flow_channels not in (2, 3):
            raise ValueError("flow_channels must be 2 or 3")
        super().__init__(num_classes, num_flow_frames, channels_per_frame=flow_channels, **options)

    def forward(self, flow: Tensor) -> Tensor:
        """Return video logits [B, num_classes] for cross-entropy training."""
        return super().forward(flow)

    def predict_video(self, flow: Tensor) -> Tensor:
        """Return probabilities; caller controls eval() and inference_mode()."""
        return self(flow).softmax(dim=1)


class DualStreamCNN(nn.Module):
    """Encode RGB and flow separately, concatenate features, classify jointly.

    RGB inputs are [B, 3, T, H, W]; flow inputs are [B, C, S, Hf, Wf]. The RGB
    stream mean-pools per-frame features over time (late fusion). The flow
    stream either stacks all S fields into its input (flow_input="stack", the
    Simonyan & Zisserman temporal stream; S is then fixed) or encodes each
    field separately and mean-pools (flow_input="mean", the original
    baseline). Both encoders have independent weights.
    """

    def __init__(self, num_classes: int = 10, *, flow_channels: int = 3,
                 normalization: str = "none", backbone: str = "small",
                 pretrained: bool = False, dropout: float = 0.0,
                 flow_input: str = "mean", num_flow_frames: int = 9):
        super().__init__()
        if type(flow_channels) is not int or flow_channels not in (2, 3):
            raise ValueError("flow_channels must be 2 or 3")
        if flow_input not in FLOW_INPUTS:
            raise ValueError(f"model.flow_input must be one of {sorted(FLOW_INPUTS)}")
        self.flow_channels = flow_channels
        self.flow_input = flow_input
        self.num_flow_frames = num_flow_frames
        options = {"normalization": normalization, "backbone": backbone,
                   "pretrained": pretrained, "dropout": dropout}
        rgb_model = PerFrameCNN(num_classes, **options)
        self.rgb_features = rgb_model.features
        self.flow_features = PerFrameCNN(num_classes, **options).features
        flow_in = flow_channels * (num_flow_frames if flow_input == "stack" else 1)
        if flow_in != 3:
            widen_input_conv(self.flow_features, flow_in, inherit=pretrained)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * rgb_model.feature_dim, num_classes)

    @staticmethod
    def _check(clips: Tensor, channels: int, name: str) -> None:
        if clips.ndim != 5 or clips.shape[1] != channels:
            raise ValueError(f"{name} must have shape [B, {channels}, T, H, W]")
        batch, _, frames, height, width = clips.shape
        if batch == 0 or frames == 0 or min(height, width) < 8:
            raise ValueError(f"{name} requires a nonempty batch/clip and height/width at least 8")

    @staticmethod
    def _encode_mean(clips: Tensor, encoder: nn.Module) -> Tensor:
        batch, channels, frames, height, width = clips.shape
        images = clips.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        return encoder(images).flatten(1).reshape(batch, frames, -1).mean(dim=1)

    def _encode_flow(self, flow: Tensor) -> Tensor:
        self._check(flow, self.flow_channels, "flow")
        if self.flow_input == "mean":
            return self._encode_mean(flow, self.flow_features)
        batch, channels, frames, height, width = flow.shape
        if frames != self.num_flow_frames:
            raise ValueError(f"Expected {self.num_flow_frames} flow fields, got {frames}")
        stacked = flow.permute(0, 2, 1, 3, 4).reshape(batch, frames * channels, height, width)
        return self.flow_features(stacked).flatten(1)

    def forward(self, rgb: Tensor, flow: Tensor) -> Tensor:
        """Return video logits [B, num_classes] for cross-entropy training."""
        self._check(rgb, 3, "rgb")
        rgb_features = self._encode_mean(rgb, self.rgb_features)
        flow_features = self._encode_flow(flow)
        if rgb.shape[0] != flow.shape[0]:
            raise ValueError("rgb and flow must have the same batch size")
        return self.classifier(self.dropout(torch.cat((rgb_features, flow_features), dim=1)))

    def predict_video(self, rgb: Tensor, flow: Tensor) -> Tensor:
        """Return probabilities; caller controls eval() and inference_mode()."""
        return self(rgb, flow).softmax(dim=1)
