"""A small 3D CNN that learns spatial and temporal features from RGB clips."""

from torch import Tensor, nn

from .normalization import normalize_convolutions


class CNN3D(nn.Module):
    """Train three spatiotemporal convolution blocks from scratch.

    Each 3x3x3 kernel slides over time, height, and width. Spatial pooling halves
    height/width while preserving the time axis, so short clips remain valid.
    Global pooling combines the final features into one vector per video.
    """

    def __init__(self, num_classes: int = 10, *, normalization: str = "none"):
        super().__init__()
        if type(num_classes) is not int or num_classes <= 0:
            raise ValueError("num_classes must be a positive integer")

        self.features = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.AdaptiveAvgPool3d(1),
        )
        self.features = normalize_convolutions(self.features, normalization)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, videos: Tensor) -> Tensor:
        """Map clips [B, 3, T, H, W] to video logits [B, num_classes].

        Inputs use the same floating-point RGB [0, 1] preprocessing as the 2D
        baselines. T must be positive and height/width must be at least 8.
        Train using cross-entropy and one target per video.
        """
        if videos.ndim != 5 or videos.shape[1] != 3:
            raise ValueError("videos must have shape [B, 3, T, H, W]")
        if videos.shape[2] == 0:
            raise ValueError("videos must contain at least one frame")
        if min(videos.shape[-2:]) < 8:
            raise ValueError("frame height and width must both be at least 8")

        # Keep time as a separate dimension throughout the convolution blocks.
        video_features = self.features(videos).flatten(1)
        return self.classifier(video_features)

    def predict_video(self, videos: Tensor) -> Tensor:
        """Return video probabilities; use eval() and inference_mode() to evaluate.

        This helper leaves training and gradient modes under caller control.
        """
        return self(videos).softmax(dim=1)
