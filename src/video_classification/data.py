"""Integrate the 2026 course datasets.py after inspecting its API and CSVs.

Canonical RGB video batches: [B, C, T, H, W]; targets: [B].
Per-frame training may use [B, C, H, W], retaining video IDs for evaluation.
Keep numeric label mappings identical across train/validation/test.
"""


def build_dataloaders(config: dict):
    """Return train/validation/test loaders using the supplied splits.

    TODO: inspect data/course/datasets.py before writing this adapter.
    Validate paths, labels, temporal ordering, and split overlaps. Randomize
    training only; apply augmentations consistently to frames in the same clip.
    For dual stream, define and validate the flow batch contract after inspecting
    its actual representation. Do not use test data to fit preprocessing.
    """
    raise NotImplementedError("Integrate the course-provided 2026 dataset loader")
