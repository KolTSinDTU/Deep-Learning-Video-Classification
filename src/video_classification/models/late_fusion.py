"""Pool shared 2D frame features over time, then classify each video."""

from torch import Tensor, nn

from .per_frame import PerFrameCNN

TEMPORAL_POOLS = {"mean", "max"}


class LateFusionCNN(nn.Module):
    """Use the per-frame CNN architecture with feature-level late fusion.

    The encoder and classifier are trained together using one label per
    video. RGB preprocessing matches PerFrameCNN. Weights start fresh unless
    training.init_checkpoint transfers a trained per-frame model.
    """

    def __init__(self, num_classes: int = 10, *, normalization: str = "none",
                 backbone: str = "small", pretrained: bool = False, dropout: float = 0.0,
                 temporal_pool: str = "mean"):
        super().__init__()
        if temporal_pool not in TEMPORAL_POOLS:
            raise ValueError(f"model.temporal_pool must be one of {sorted(TEMPORAL_POOLS)}")
        frame_model = PerFrameCNN(num_classes=num_classes, normalization=normalization,
                                  backbone=backbone, pretrained=pretrained, dropout=dropout)
        self.temporal_pool = temporal_pool
        self.features = frame_model.features
        self.dropout = frame_model.dropout
        self.classifier = frame_model.classifier

    def forward(self, videos: Tensor) -> Tensor:
        """Map clips [B, 3, T, H, W] to video logits [B, num_classes].

        The same encoder processes every frame. Temporal mean/max pooling
        ignores frame order and supports any positive number of frames. Train
        with cross-entropy on these logits and one target per video.
        """
        if videos.ndim != 5 or videos.shape[1] != 3:
            raise ValueError("videos must have shape [B, 3, T, H, W]")
        batch_size, channels, num_frames, height, width = videos.shape
        if num_frames == 0:
            raise ValueError("videos must contain at least one frame")
        if min(height, width) < 8:
            raise ValueError("frame height and width must both be at least 8")

        # Group time with batch while keeping each RGB frame intact.
        frames = videos.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_frames, channels, height, width
        )
        frame_features = self.features(frames).flatten(1)
        frame_features = frame_features.reshape(batch_size, num_frames, -1)
        video_features = (frame_features.mean(dim=1) if self.temporal_pool == "mean"
                          else frame_features.amax(dim=1))
        return self.classifier(self.dropout(video_features))

    def predict_video(self, videos: Tensor) -> Tensor:
        """Return video probabilities; softmax is applied AFTER feature fusion.

        For inference, call eval() and use torch.inference_mode(). This helper
        leaves training and gradient modes under caller control.
        """
        return self(videos).softmax(dim=1)
