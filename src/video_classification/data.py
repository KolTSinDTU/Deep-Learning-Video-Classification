"""Load the course CSV/frame layout with portable paths and validated labels.

This adapts the frame/video contracts in the supplied src/datasets.py. Using
pathlib and CSV-indexed frames avoids its cluster paths and slash assumptions.
The supplied file is retained as a reference. Test data is not loaded here.
"""

import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


class ActionDataset(Dataset):
    """Return individual RGB frames or clips [3, T, H, W], with video labels."""

    def __init__(self, root: Path, split: str, num_frames: int, image_size: int,
                 *, per_frame: bool):
        self.per_frame = per_frame
        self.num_frames = num_frames
        self.transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])
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
                directory = Path(root) / "frames" / split / relative.with_suffix("")
                paths = tuple(directory / f"frame_{index}.jpg"
                              for index in range(1, num_frames + 1))
                for path in paths:
                    if not path.is_file():
                        raise FileNotFoundError(f"Missing {split} frame: {path}")
                self.records.append((video_id, label, paths))
                self.video_ids.add(video_id)
                self.class_to_label[action] = label
                label_to_class[label] = action
        if not self.records:
            raise ValueError(f"No videos found in {metadata}")

    def __len__(self):
        return len(self.records) * (self.num_frames if self.per_frame else 1)

    def _load_frame(self, path):
        with Image.open(path) as image:
            return self.transform(image.convert("RGB"))

    def __getitem__(self, index):
        if self.per_frame:
            video_index, frame_index = divmod(index, self.num_frames)
            _, label, paths = self.records[video_index]
            return self._load_frame(paths[frame_index]), label
        _, label, paths = self.records[index]
        return torch.stack([self._load_frame(path) for path in paths], dim=1), label


def _seed_worker(worker_id):
    """Seed Python/NumPy from the seed PyTorch assigns to this worker."""
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def build_dataloaders(config: dict) -> dict[str, DataLoader]:
    """Return shuffled training frames and ordered validation videos.

    Keep the supplied splits and numeric labels. Exact video overlap is checked;
    this is not a subject-identity audit and does not fix the original leakage.
    Only train/val metadata and frames are accessed.
    """
    if config["model"]["name"] != "per_frame":
        raise NotImplementedError("Data loading currently supports the per_frame model")
    data, training = config["dataset"], config["training"]
    datasets = {
        split: ActionDataset(Path(data["root"]), split, data["num_frames"],
                             data["image_size"], per_frame=(split == "train"))
        for split in ("train", "val")
    }
    mapping = datasets["train"].class_to_label
    if sorted(mapping.values()) != list(range(data["num_classes"])):
        raise ValueError("Training labels must cover 0 through dataset.num_classes - 1")
    for action, label in datasets["val"].class_to_label.items():
        if mapping.get(action) != label:
            raise ValueError(f"Validation class/label disagrees with training: {action}={label}")
    overlap = datasets["train"].video_ids & datasets["val"].video_ids
    if overlap:
        raise ValueError(f"Videos occur in both train and validation: {sorted(overlap)[:5]}")

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
