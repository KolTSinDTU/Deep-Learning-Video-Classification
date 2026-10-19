import os
from glob import glob
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms as T

# Dataset paths
DATASET_PATHS = {
    "leakage": "/dtu/datasets1/02516/ufc10",
    "no_leakage": "/dtu/datasets1/02516/ucf101_noleakage",
}


def get_data_root(dataset_name="no_leakage"):
    """Get the root directory for a dataset."""
    if dataset_name not in DATASET_PATHS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Choose from {list(DATASET_PATHS.keys())}"
        )
    return DATASET_PATHS[dataset_name]


class FrameImageDataset(torch.utils.data.Dataset):
    """Dataset that returns individual frames from videos."""

    def __init__(self, dataset_name="no_leakage", split="train", transform=None):
        root_dir = get_data_root(dataset_name)
        self.frame_paths = sorted(glob(f"{root_dir}/frames/{split}/*/*/*.jpg"))
        self.df = pd.read_csv(f"{root_dir}/metadata/{split}.csv")
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.frame_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        video_name = frame_path.split("/")[-2]
        video_meta = self._get_meta("video_name", video_name)
        label = video_meta["label"].item()

        frame = Image.open(frame_path).convert("RGB")

        if self.transform:
            frame = self.transform(frame)
        else:
            frame = T.ToTensor()(frame)

        return frame, label


class FrameVideoDataset(torch.utils.data.Dataset):
    """Dataset that returns all frames from a video."""

    def __init__(
        self,
        dataset_name="no_leakage",
        split="train",
        transform=None,
        stack_frames=True,
    ):
        root_dir = get_data_root(dataset_name)
        self.video_paths = sorted(glob(f"{root_dir}/videos/{split}/*/*.avi"))
        self.df = pd.read_csv(f"{root_dir}/metadata/{split}.csv")
        self.split = split
        self.transform = transform
        self.stack_frames = stack_frames
        self.n_sampled_frames = 10

    def __len__(self):
        return len(self.video_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = video_path.split("/")[-1].split(".avi")[0]
        video_meta = self._get_meta("video_name", video_name)
        label = video_meta["label"].item()

        video_frames_dir = video_path.replace(".avi", "").replace("videos", "frames")
        video_frames = self.load_frames(video_frames_dir)

        if self.transform:
            frames = [self.transform(frame) for frame in video_frames]
        else:
            frames = [T.ToTensor()(frame) for frame in video_frames]

        if self.stack_frames:
            frames = torch.stack(frames).permute(1, 0, 2, 3)  # [C, T, H, W]

        return frames, label

    def load_frames(self, frames_dir):
        frames = []
        for i in range(1, self.n_sampled_frames + 1):
            frame_file = os.path.join(frames_dir, f"frame_{i}.jpg")
            frame = Image.open(frame_file).convert("RGB")
            frames.append(frame)
        return frames
