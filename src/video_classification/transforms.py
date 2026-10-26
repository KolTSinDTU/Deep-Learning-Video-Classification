"""Clip-consistent spatial augmentation and input normalization.

One set of random parameters is sampled per clip and applied to every RGB
frame and flow field in it, so augmentation never invents motion between
frames. Legacy settings (no augmentation, no normalization) reproduce the
original ToTensor-only preprocessing exactly.
"""

import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as TF

INPUT_NORMALIZATIONS = {"none", "standardize"}
# ImageNet statistics; fixed constants, so nothing is fitted on validation/test data.
RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)
CROP_SCALE = (0.5, 1.0)
CROP_RATIO = (3 / 4, 4 / 3)
JITTER = {"brightness": (0.8, 1.2), "contrast": (0.8, 1.2), "saturation": (0.8, 1.2)}


class ClipTransform:
    """Resize/crop/flip/jitter a clip identically across its frames.

    Training with augmentation uses a random resized crop to image_size, a
    horizontal flip with probability 0.5 (when allowed), and colour jitter on
    RGB only. Otherwise frames are resized to image_size (or kept native).
    """

    def __init__(self, image_size: int | str, *, augment: bool = False,
                 normalization: str = "none", allow_flip: bool = True):
        if normalization not in INPUT_NORMALIZATIONS:
            raise ValueError(f"dataset.input_normalization must be one of {sorted(INPUT_NORMALIZATIONS)}")
        if augment and image_size == "native":
            raise ValueError("dataset.augment requires an integer dataset.image_size")
        self.image_size = image_size
        self.augment = augment
        self.normalization = normalization
        self.allow_flip = allow_flip

    def sample(self, image: Image.Image) -> dict:
        """Draw one set of clip parameters using torch's (worker-seeded) RNG."""
        if not self.augment:
            return {"crop": None, "flip": False, "jitter": None}
        crop = T.RandomResizedCrop.get_params(image, CROP_SCALE, CROP_RATIO)
        flip = self.allow_flip and bool(torch.rand(()) < 0.5)
        jitter = T.ColorJitter.get_params(JITTER["brightness"], JITTER["contrast"],
                                          JITTER["saturation"], None)
        return {"crop": crop, "flip": flip, "jitter": jitter}

    def _spatial(self, image, params):
        if params["crop"] is not None:
            top, left, height, width = params["crop"]
            image = TF.resized_crop(image, top, left, height, width,
                                    [self.image_size, self.image_size], antialias=True)
        elif self.image_size != "native":
            image = TF.resize(image, [self.image_size, self.image_size], antialias=True)
        return TF.hflip(image) if params["flip"] else image

    def rgb(self, image: Image.Image, params: dict) -> torch.Tensor:
        """Return a [3, H, W] float frame."""
        image = self._spatial(image.convert("RGB"), params)
        if params["jitter"] is not None:
            order, brightness, contrast, saturation, _ = params["jitter"]
            adjust = {0: lambda x: TF.adjust_brightness(x, brightness),
                      1: lambda x: TF.adjust_contrast(x, contrast),
                      2: lambda x: TF.adjust_saturation(x, saturation)}
            for index in order.tolist():
                if index in adjust:
                    image = adjust[index](image)
        tensor = TF.to_tensor(image)
        if self.normalization == "standardize":
            tensor = TF.normalize(tensor, RGB_MEAN, RGB_STD)
        return tensor

    def flow_png(self, image: Image.Image, params: dict) -> torch.Tensor:
        """Return a [3, H, W] colour-coded flow field.

        Colour-wheel flow cannot be mirrored by flipping pixels (hue encodes
        direction), so callers must disable flips for this format. With
        standardization, values are inverted so zero motion (white) becomes 0.
        """
        if params["flip"]:
            raise ValueError("Horizontal flips are not valid for colour-coded flow")
        tensor = TF.to_tensor(self._spatial(image, params))
        return 1.0 - tensor if self.normalization == "standardize" else tensor

    def flow_uv(self, flow: torch.Tensor, params: dict) -> torch.Tensor:
        """Resize/crop/flip a [2, H, W] pixel displacement field and its vectors."""
        _, height, width = flow.shape
        if params["crop"] is not None:
            _, _, height, width = params["crop"]
        flow = self._spatial(flow, params)
        _, new_height, new_width = flow.shape
        flow = flow * flow.new_tensor([new_width / width, new_height / height])[:, None, None]
        if params["flip"]:
            flow = flow * flow.new_tensor([-1.0, 1.0])[:, None, None]
        return flow
