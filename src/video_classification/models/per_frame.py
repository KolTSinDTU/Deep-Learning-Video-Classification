"""A small 2D CNN trained on frames, with probability averaging for videos."""

from torch import Tensor, nn

from .normalization import normalize_convolutions


class PerFrameCNN(nn.Module):
    """Classify RGB frames independently using a CNN trained from scratch.

    Inputs must be floating-point RGB tensors, normally scaled to [0, 1],
    with height and width of at least 8 pixels. Preprocessing belongs in the
    data loader. Adaptive pooling allows different spatial resolutions.
    """

    def __init__(self, num_classes: int = 10, *, normalization: str = "none"):
        super().__init__()
        if type(num_classes) is not int or num_classes <= 0:
            raise ValueError("num_classes must be a positive integer")

        self.features = nn.Sequential(
            # Each block learns spatial features, then halves the resolution.
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # One value per feature channel, independent of input resolution.
            nn.AdaptiveAvgPool2d(1),
        )
        self.features = normalize_convolutions(self.features, normalization)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, frames: Tensor) -> Tensor:
        """Map frames [B, 3, H, W] to raw logits [B, num_classes].

        Train with CrossEntropyLoss(logits, labels), using each frame's video
        label. Do not apply softmax before that loss.
        """
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError("frames must have shape [B, 3, H, W]")
        if min(frames.shape[-2:]) < 8:
            raise ValueError("frame height and width must both be at least 8")
        features = self.features(frames).flatten(1)
        return self.classifier(features)

    def predict_video(self, videos: Tensor) -> Tensor:
        """Average frame probabilities for clips [B, 3, T, H, W].

        Returns probabilities [B, num_classes], not logits. Softmax is applied
        to each frame BEFORE averaging over time. Use argmax(dim=1) for one
        predicted label per video. This baseline ignores temporal order.

        For evaluation, call model.eval() and use torch.inference_mode().
        This method does not change the model's training or gradient mode.
        """
        if videos.ndim != 5 or videos.shape[1] != 3:
            raise ValueError("videos must have shape [B, 3, T, H, W]")
        batch_size, channels, num_frames, height, width = videos.shape
        if num_frames == 0:
            raise ValueError("videos must contain at least one frame")

        # Put time next to batch before flattening, keeping each RGB frame intact.
        frames = videos.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_frames, channels, height, width
        )
        frame_logits = self(frames).reshape(batch_size, num_frames, -1)
        return frame_logits.softmax(dim=-1).mean(dim=1)
