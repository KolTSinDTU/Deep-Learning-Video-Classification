"""Explicitly pair consecutive RGB frames with color or vector optical flow."""

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .config import validate_dual_stream
from .data import ActionDataset


class DualStreamDataset(ActionDataset):
    """Return {rgb: [3,T,H,W], flow: [C,T-1,H,W]} and a video label.

    streams=("flow",) returns only {flow: ...} for the flow-only temporal
    stream. RGB and flow share one sampled crop. Colour-coded flow cannot be
    mirrored, so horizontal flips are disabled for rgb_png.

    The course layout names each pair as flows_png/<split>/<class>/<video>/
    flow_<from>_<to>.png. Optional manifests support custom representations;
    native caches require a manifest with matching source-frame indices.
    """

    def __init__(self, data: dict, split: str, *, augment: bool = False,
                 streams: tuple[str, ...] = ("rgb", "flow")):
        validate_dual_stream(data)
        if not streams or not set(streams) <= {"rgb", "flow"} or "flow" not in streams:
            raise ValueError("streams must be ('flow',) or ('rgb', 'flow')")
        root = Path(data["root"]).resolve()
        source = data.get("frame_source", "provided")
        super().__init__(root, split, data["num_frames"], data["image_size"],
                         per_frame=False, frame_source=source,
                         frame_cache_root=data.get("frame_cache_root"), augment=augment,
                         input_normalization=data.get("input_normalization", "none"),
                         allow_flip=(data["flow_format"] != "rgb_png"))
        self.streams = tuple(streams)
        self.flow_format = data["flow_format"]
        self.flow_scale = data.get("flow_scale", 1.0)
        self.image_size = data["image_size"]
        required = {"video_name", "from_frame", "to_frame", "flow_path"}
        if source == "native_video":
            required |= {"source_from_index", "source_to_index"}
        if "flow_manifest_root" in data:
            manifest = Path(data["flow_manifest_root"]) / f"{split}.csv"
            with manifest.open(newline="", encoding="utf-8-sig") as stream:
                reader = csv.DictReader(stream)
                if not required.issubset(reader.fieldnames or []):
                    raise ValueError(f"{manifest} must have columns {sorted(required)}")
                rows = list(reader)
        else:
            if source != "provided" or self.flow_format != "rgb_png":
                raise ValueError("Automatic course flow loading requires provided RGB frames and "
                                 "rgb_png; custom/native flow needs dataset.flow_manifest_root")
            rows = []
            for video_id, _, rgb_paths in self.records:
                # RGB paths already follow the validated video_path from metadata.
                relative_video = rgb_paths[0].parent.relative_to(root / "frames" / split)
                for start in range(1, self.num_frames):
                    relative = Path("flows_png") / split / relative_video / f"flow_{start}_{start+1}.png"
                    rows.append({"video_name": video_id, "from_frame": start,
                                 "to_frame": start + 1, "flow_path": relative.as_posix()})
        pairs = {video_id: {} for video_id in self.video_ids}
        self.flow_paths = set()
        for row in rows:
            video_id = row["video_name"]
            if video_id not in pairs:
                raise ValueError(f"Flow manifest contains unknown {split} video: {video_id}")
            start, end = int(row["from_frame"]), int(row["to_frame"])
            if not 1 <= start < self.num_frames or end != start + 1:
                raise ValueError(f"Flow must map consecutive RGB frames in 1..{self.num_frames}")
            if start in pairs[video_id]:
                raise ValueError(f"Duplicate flow pair for {video_id}: {start}->{end}")
            relative = Path(row["flow_path"].replace("\\", "/"))
            path = (root / relative).resolve()
            if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root):
                raise ValueError(f"flow_path must stay within dataset.root: {relative}")
            extension = ".png" if self.flow_format == "rgb_png" else ".npy"
            if path.suffix.lower() != extension:
                raise ValueError(f"{self.flow_format} requires {extension} files: {path}")
            if not path.is_file():
                raise FileNotFoundError(f"Missing {split} flow: {path}")
            if path in self.flow_paths:
                raise ValueError(f"Flow file reused for different frame pairs: {path}")
            self.flow_paths.add(path)
            pairs[video_id][start] = (path, row)
        self.flow_records = []
        for video_id, _, rgb_paths in self.records:
            if set(pairs[video_id]) != set(range(1, self.num_frames)):
                raise ValueError(f"Missing consecutive flow pairs for {video_id}")
            ordered = [pairs[video_id][i] for i in range(1, self.num_frames)]
            if source == "native_video":
                sampling = json.loads((rgb_paths[0].parent / "sampling.json").read_text())
                indices = sampling.get("frame_indices_zero_based")
                if not isinstance(indices, list) or len(indices) != self.num_frames:
                    raise ValueError(f"Native cache needs source frame indices: {video_id}")
                for i, (_, row) in enumerate(ordered):
                    if (int(row["source_from_index"]), int(row["source_to_index"])) != tuple(indices[i:i+2]):
                        raise ValueError(f"Flow source indices disagree with native RGB cache: {video_id}")
            self.flow_records.append(tuple(path for path, _ in ordered))

    def _load_flow(self, path: Path, expected_size: tuple[int, int], params: dict) -> torch.Tensor:
        if self.flow_format == "rgb_png":
            with Image.open(path) as image:
                if image.format != "PNG" or image.mode != "RGB":
                    raise ValueError(f"rgb_png requires an RGB color PNG, not encoded vectors: {path}")
                if image.size != expected_size:
                    raise ValueError(f"Flow/RGB spatial dimensions disagree: {path}")
                return self.clip_transform.flow_png(image, params)
        values = np.load(path, allow_pickle=False)
        if (values.ndim != 3 or values.shape[-1] != 2
                or not np.issubdtype(values.dtype, np.floating)
                or not np.isfinite(values).all()):
            raise ValueError(f"uv_npy requires finite floating-point [H, W, 2] pixel displacements: {path}")
        height, width, _ = values.shape
        if (width, height) != expected_size:
            raise ValueError(f"Flow/RGB spatial dimensions disagree: {path}")
        flow = torch.from_numpy(values.astype(np.float32)).permute(2, 0, 1)
        # Displacements are in pixels: crop/resize/flip both the field and its vectors.
        flow = self.clip_transform.flow_uv(flow, params) / self.flow_scale
        if not torch.isfinite(flow).all():
            raise ValueError(f"Flow preprocessing produced non-finite values: {path}")
        return flow

    def __getitem__(self, index):
        _, label, rgb_paths = self.records[index]
        sizes = []
        for path in rgb_paths:
            # Opening reads only the header; pixels are decoded below if needed.
            with Image.open(path) as image:
                sizes.append(image.size)
        if len(set(sizes)) != 1:
            raise ValueError("RGB frames in a paired clip must have identical source dimensions")
        sample = {}
        if "rgb" in self.streams:
            sample["rgb"], params = self._load_clip(rgb_paths)
        else:
            with Image.open(rgb_paths[0]) as image:
                params = self.clip_transform.sample(image)
        flow = [self._load_flow(path, sizes[0], params) for path in self.flow_records[index]]
        sample["flow"] = torch.stack(flow, dim=1)
        return sample, label
