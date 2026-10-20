"""Stack RGB frames into channels before applying the 2D CNN."""

from torch import Tensor, nn

from .per_frame import PerFrameCNN


class EarlyFusionCNN(nn.Module):
    """Use the per-frame CNN architecture with a wider input convolution.

    Each video becomes a 3*num_frames-channel image in temporal RGB order:
    R1, G1, B1, R2, G2, B2, ... . The first convolution can combine frames
    immediately. Frame order matters and the frame count is fixed at creation.
    All layers, including the wider first convolution, use fresh random weights.
    """

    def __init__(self, num_classes: int = 10, num_frames: int = 10,
                 *, normalization: str = "none"):
        super().__init__()
        if type(num_frames) is not int or num_frames <= 0:
            raise ValueError("num_frames must be a positive integer")
        self.num_frames = num_frames
        frame_model = PerFrameCNN(num_classes=num_classes, normalization=normalization)
        self.features = frame_model.features
        first_conv = self.features[0]
        self.features[0] = nn.Conv2d(
            3 * num_frames, first_conv.out_channels,
            kernel_size=first_conv.kernel_size, stride=first_conv.stride,
            padding=first_conv.padding, bias=(first_conv.bias is not None),
        )
        self.classifier = frame_model.classifier

    def forward(self, videos: Tensor) -> Tensor:
        """Map [B, 3, T, H, W] clips to [B, num_classes] video logits.

        Use floating-point RGB inputs scaled to [0, 1], matching the per-frame
        baseline. Train with cross-entropy and one target per video.
        """
        if videos.ndim != 5 or videos.shape[1] != 3:
            raise ValueError("videos must have shape [B, 3, T, H, W]")
        batch_size, channels, num_frames, height, width = videos.shape
        if num_frames != self.num_frames:
            raise ValueError(f"Expected {self.num_frames} frames per video, got {num_frames}")
        if min(height, width) < 8:
            raise ValueError("frame height and width must both be at least 8")

        # Keep each frame's RGB channels together, then fuse time into channels.
        stacked_frames = videos.permute(0, 2, 1, 3, 4).reshape(
            batch_size, num_frames * channels, height, width
        )
        video_features = self.features(stacked_frames).flatten(1)
        return self.classifier(video_features)

    def predict_video(self, videos: Tensor) -> Tensor:
        """Return video probabilities; use eval() and inference_mode() to evaluate.

        This method leaves training and gradient modes under caller control.
        """
        return self(videos).softmax(dim=1)
