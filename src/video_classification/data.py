"""Load the course CSV/frame layout with portable paths and validated labels.

This adapts the frame/video contracts in the supplied src/datasets.py. Using
pathlib and CSV-indexed frames avoids its cluster paths and slash assumptions.
The supplied file is retained as a reference. Test data is not loaded here.
"""

import csv
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from .config import MODEL_NAMES, validate_image_size
from .transforms import ClipTransform


class ActionDataset(Dataset):
    """Return individual RGB frames or clips [3, T, H, W], with video labels."""

    def __init__(self, root: Path, split: str, num_frames: int, image_size: int | str,
                 *, per_frame: bool, frame_source: str = "provided",
                 frame_cache_root: str | Path | None = None, augment: bool = False,
                 input_normalization: str = "none", allow_flip: bool = True):
        self.per_frame = per_frame
        self.num_frames = num_frames
        validate_image_size(image_size)
        if frame_source not in {"provided", "native_video"}:
            raise ValueError("frame_source must be provided or native_video")
        frame_directory = "frames_native" if frame_source == "native_video" else "frames"
        frames_root = (Path(frame_cache_root) if frame_source == "native_video" and frame_cache_root
                       else Path(root) / frame_directory)
        extension = "png" if frame_source == "native_video" else "jpg"
        # Native mode preserves every source pixel and the original aspect ratio.
        # Without augmentation/normalization this is ToTensor (uint8 / 255) only.
        self.clip_transform = ClipTransform(image_size, augment=augment,
                                            normalization=input_normalization,
                                            allow_flip=allow_flip)
        self.records = []
        self.video_ids = set()
        self.class_to_label = {}
        label_to_class = {}
        metadata = Path(root) / "metadata" / f"{split}.csv"
        with metadata.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            required = {"video_name", "video_path", "action", "label"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"{metadata} must have columns {sorted(required)}")
            for row in reader:
                video_id, action, label = row["video_name"], row["action"], int(row["label"])
                if video_id in self.video_ids:
                    raise ValueError(f"Duplicate video in {split}: {video_id}")
                if not action or label < 0:
                    raise ValueError(f"Invalid class/label in {metadata}: {row}")
                if (self.class_to_label.get(action, label) != label
                        or label_to_class.get(label, action) != action):
                    raise ValueError(f"Inconsistent class-to-label mapping in {metadata}")
                # CSV video_path is relative to videos/<split>; frames mirror it.
                relative = Path(row["video_path"].replace("\\", "/"))
                if relative.is_absolute() or ".." in relative.parts or relative.stem != video_id:
                    raise ValueError(f"Invalid video_path for {video_id}: {relative}")
                directory = frames_root / split / relative.with_suffix("")
                paths = tuple(directory / f"frame_{index}.{extension}"
                              for index in range(1, num_frames + 1))
                for path in paths:
                    if not path.is_file():
                        hint = ("; run python -m video_classification.prepare_frames --root "
                                f'"{root}" --output-root "{frames_root}" '
                                f'--splits {split} --num-frames {num_frames}'
                                if frame_source == "native_video" else "")
                        raise FileNotFoundError(f"Missing {split} frame: {path}{hint}")
                if frame_source == "native_video":
                    manifest = json.loads((directory / "sampling.json").read_text(encoding="utf-8"))
                    if manifest["num_frames"] != num_frames:
                        raise ValueError(f"Native cache frame count disagrees with config: {directory}")
                self.records.append((video_id, label, paths))
                self.video_ids.add(video_id)
                self.class_to_label[action] = label
                label_to_class[label] = action
        if not self.records:
            raise ValueError(f"No videos found in {metadata}")

    def __len__(self):
        return len(self.records) * (self.num_frames if self.per_frame else 1)

    def _load_clip(self, paths):
        """Load frames [3, T, H, W] using one set of spatial parameters."""
        frames, params = [], None
        for path in paths:
            with Image.open(path) as image:
                if params is None:
                    params = self.clip_transform.sample(image)
                frames.append(self.clip_transform.rgb(image, params))
        return torch.stack(frames, dim=1), params

    def __getitem__(self, index):
        if self.per_frame:
            video_index, frame_index = divmod(index, self.num_frames)
            _, label, paths = self.records[video_index]
            return self._load_clip(paths[frame_index:frame_index + 1])[0][:, 0], label
        _, label, paths = self.records[index]
        return self._load_clip(paths)[0], label


def _seed_worker(worker_id):
    """Seed Python/NumPy from the seed PyTorch assigns to this worker."""
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def build_dataset(config: dict, split: str, *, train: bool):
    """Construct one split; only training splits are augmented."""
    data, model_name = config["dataset"], config["model"]["name"]
    augment = train and data.get("augment", False)
    if model_name in {"dual_stream", "flow_stream"}:
        from .flow_data import DualStreamDataset

        streams = ("flow",) if model_name == "flow_stream" else ("rgb", "flow")
        return DualStreamDataset(data, split, augment=augment, streams=streams)
    return ActionDataset(Path(data["root"]), split, data["num_frames"], data["image_size"],
                         per_frame=(train and model_name == "per_frame"),
                         frame_source=data.get("frame_source", "provided"),
                         frame_cache_root=data.get("frame_cache_root"), augment=augment,
                         input_normalization=data.get("input_normalization", "none"))


def build_eval_loader(config: dict, split: str) -> DataLoader:
    """Return ordered whole-video batches for val or (final) test evaluation."""
    if split not in {"val", "test"}:
        raise ValueError("split must be val or test")
    return DataLoader(build_dataset(config, split, train=False),
                      batch_size=config["training"].get("validation_batch_size", 1),
                      shuffle=False, num_workers=config["training"]["num_workers"])


def build_dataloaders(config: dict) -> dict[str, DataLoader]:
    """Return shuffled training samples and ordered validation videos.

    Per-frame training uses images; fusion and 3D models use whole videos.
    Keep the supplied splits and numeric labels. Exact video overlap is checked;
    this is not a subject-identity audit and does not fix the original leakage.
    Only train/val metadata and frames are accessed.
    """
    model_name = config["model"]["name"]
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {model_name}")
    data, training = config["dataset"], config["training"]
    datasets = {split: build_dataset(config, split, train=(split == "train"))
                for split in ("train", "val")}
    mapping = datasets["train"].class_to_label
    if sorted(mapping.values()) != list(range(data["num_classes"])):
        raise ValueError("Training labels must cover 0 through dataset.num_classes - 1")
    for action, label in datasets["val"].class_to_label.items():
        if mapping.get(action) != label:
            raise ValueError(f"Validation class/label disagrees with training: {action}={label}")
    overlap = datasets["train"].video_ids & datasets["val"].video_ids
    if overlap:
        raise ValueError(f"Videos occur in both train and validation: {sorted(overlap)[:5]}")
    if model_name in {"dual_stream", "flow_stream"} and datasets["train"].flow_paths & datasets["val"].flow_paths:
        raise ValueError("Flow files occur in both train and validation")

    return {
        split: DataLoader(
            dataset,
            batch_size=(training["batch_size"] if split == "train"
                        else training.get("validation_batch_size", 1)),
            shuffle=(split == "train"),
            num_workers=training["num_workers"],
            worker_init_fn=_seed_worker,
            generator=torch.Generator().manual_seed(config["experiment"]["seed"]),
            pin_memory=(training.get("device") == "cuda"),
        )
        for split, dataset in datasets.items()
    }
