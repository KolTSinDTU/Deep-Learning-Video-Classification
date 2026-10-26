"""A 3D CNN that learns spatial and temporal features from RGB clips."""

from torch import Tensor, nn

from .backbone import build_backbone


class CNN3D(nn.Module):
    """Spatiotemporal convolutions over time, height, and width.

    The small backbone has three 3x3x3 blocks and pools space only. The
    vgg_bn backbone mirrors the 2D vgg_bn layer-for-layer (so a trained
    per-frame model can be inflated into it, I3D-style) and halves time in
    its last two stages, giving every feature a receptive field spanning the
    whole 10-frame clip. Global pooling yields one vector per video.
    """

    def __init__(self, num_classes: int = 10, *, normalization: str = "none",
                 backbone: str = "small", dropout: float = 0.0):
        super().__init__()
        if type(num_classes) is not int or num_classes <= 0:
            raise ValueError("num_classes must be a positive integer")
        if not 0 <= dropout < 1:
            raise ValueError("model.dropout must be in [0, 1)")
        self.backbone = backbone
        self.features, feature_dim = build_backbone(backbone, dimensions=3,
                                                    normalization=normalization)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feature_dim, num_classes)

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
        if min(videos.shape[-2:]) < (16 if self.backbone == "vgg_bn" else 8):
            raise ValueError("frame height and width must both be at least 8 (16 for vgg_bn)")
        if self.backbone == "vgg_bn" and videos.shape[2] < 4:
            raise ValueError("the vgg_bn 3D backbone pools time twice; use at least 4 frames")

        # Keep time as a separate dimension throughout the convolution blocks.
        video_features = self.features(videos).flatten(1)
        return self.classifier(self.dropout(video_features))

    def predict_video(self, videos: Tensor) -> Tensor:
        """Return video probabilities; use eval() and inference_mode() to evaluate.

        This helper leaves training and gradient modes under caller control.
        """
        return self(videos).softmax(dim=1)
