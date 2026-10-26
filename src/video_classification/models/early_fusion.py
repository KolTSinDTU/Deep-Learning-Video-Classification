"""Stack frames (RGB or flow) into channels before applying the 2D CNN."""

from torch import Tensor, nn

from .backbone import widen_input_conv
from .per_frame import PerFrameCNN


class EarlyFusionCNN(nn.Module):
    """Use the per-frame CNN architecture with a wider input convolution.

    Each clip becomes a channels_per_frame*num_frames-channel image in temporal
    order (R1, G1, B1, R2, G2, B2, ... for RGB). The first convolution can
    combine frames immediately. Frame order matters and the frame count is
    fixed at creation. With a pretrained backbone the wider input kernel is
    derived from the RGB kernel; otherwise it starts from fresh random weights
    (training.init_checkpoint can still inflate a trained per-frame kernel).
    """

    def __init__(self, num_classes: int = 10, num_frames: int = 10,
                 *, normalization: str = "none", backbone: str = "small",
                 pretrained: bool = False, dropout: float = 0.0, channels_per_frame: int = 3):
        super().__init__()
        if type(num_frames) is not int or num_frames <= 0:
            raise ValueError("num_frames must be a positive integer")
        if type(channels_per_frame) is not int or channels_per_frame <= 0:
            raise ValueError("channels_per_frame must be a positive integer")
        self.num_frames = num_frames
        self.channels_per_frame = channels_per_frame
        frame_model = PerFrameCNN(num_classes=num_classes, normalization=normalization,
                                  backbone=backbone, pretrained=pretrained, dropout=dropout)
        self.features = frame_model.features
        widen_input_conv(self.features, channels_per_frame * num_frames, inherit=pretrained)
        self.dropout = frame_model.dropout
        self.classifier = frame_model.classifier

    def forward(self, videos: Tensor) -> Tensor:
        """Map [B, C, T, H, W] clips to [B, num_classes] video logits.

        Inputs use the same preprocessing as the matching single-frame model.
        Train with cross-entropy and one target per video.
        """
        channels = self.channels_per_frame
        if videos.ndim != 5 or videos.shape[1] != channels:
            raise ValueError(f"videos must have shape [B, {channels}, T, H, W]")
        batch_size, channels, num_frames, height, width = videos.shape
        if num_frames != self.num_frames:
            raise ValueError(f"Expected {self.num_frames} frames per video, got {num_frames}")
        if min(height, width) < 8:
            raise ValueError("frame height and width must both be at least 8")

        # Keep each frame's channels together, then fuse time into channels.
        stacked_frames = videos.permute(0, 2, 1, 3, 4).reshape(
            batch_size, num_frames * channels, height, width
        )
        video_features = self.features(stacked_frames).flatten(1)
        return self.classifier(self.dropout(video_features))

    def predict_video(self, videos: Tensor) -> Tensor:
        """Return video probabilities; use eval() and inference_mode() to evaluate.

        This method leaves training and gradient modes under caller control.
        """
        return self(videos).softmax(dim=1)
