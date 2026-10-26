"""Cache evenly spaced, full-resolution RGB frames from source videos.

Requires ffmpeg and ffprobe on PATH. Writes lossless PNGs to frames_native;
the course-provided JPEGs and source videos are never modified. Only explicitly
selected splits are read (train and val by default).
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

from PIL import Image


def sample_indices(total: int, count: int) -> list[int]:
    """Include both endpoints, using floor-spaced zero-based frame indices."""
    if count <= 0 or total < count:
        raise ValueError(f"Need at least {count} source frames; found {total}")
    if count == 1:
        return [total // 2]
    return [index * (total - 1) // (count - 1) for index in range(count)]


def _run(arguments: list[str]) -> str:
    result = subprocess.run(arguments, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"{arguments[0]} failed: {result.stderr.strip()}")
    return result.stdout


def extract_video(source: Path, destination: Path, count: int) -> str:
    """Publish a complete cache atomically; reuse only a matching manifest."""
    stat = source.stat()
    identity = {"source_size": stat.st_size, "source_mtime_ns": stat.st_mtime_ns,
                "num_frames": count, "sampling": "uniform_floor_endpoints_v1"}
    if destination.exists():
        manifest = json.loads((destination / "sampling.json").read_text(encoding="utf-8"))
        if (all(manifest.get(key) == value for key, value in identity.items())
                and all((destination / f"frame_{i}.png").is_file() for i in range(1, count + 1))):
            return "reused"
        raise ValueError(f"Cache does not match source/settings: {destination}; use a fresh cache location")

    probe = json.loads(_run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=width,height,nb_read_frames", "-of", "json", str(source),
    ]))["streams"][0]
    # AVI header counts can exceed the number of decodable frames by one.
    total = probe["nb_read_frames"]
    indices = sample_indices(int(total), count)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".extract_", dir=destination.parent) as temporary:
        staging = Path(temporary) / "frames"
        staging.mkdir()
        selection = "+".join(f"eq(n\\,{index})" for index in indices)
        _run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
            "-threads", "1", "-i", str(source), "-map", "0:v:0",
            "-vf", f"select={selection}", "-fps_mode", "passthrough", "-frames:v", str(count),
            "-pix_fmt", "rgb24", "-threads", "1", str(staging / "frame_%d.png"),
        ])
        for index in range(1, count + 1):
            if not (staging / f"frame_{index}.png").is_file():
                raise ValueError(f"Could not decode sample {index}/{count} from {source}")
            with Image.open(staging / f"frame_{index}.png") as frame:
                if frame.size != (probe["width"], probe["height"]):
                    raise ValueError(f"Decoded dimensions differ from source: {source}")
                frame.verify()
        manifest = {**identity, "source": str(source), "width": probe["width"],
                    "height": probe["height"], "source_frame_count": int(total),
                    "frame_indices_zero_based": indices}
        (staging / "sampling.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staging.rename(destination)
    return "created"


def prepare(root: Path, splits: list[str], count: int, workers: int,
            output_root: Path | None = None) -> None:
    if count <= 0 or workers <= 0:
        raise ValueError("num-frames and workers must be positive")
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise ValueError(f"{executable} must be installed and available on PATH")
    output_root = output_root if output_root is not None else root / "frames_native"
    jobs = []
    seen = set()
    for split in splits:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {split}")
        with (root / "metadata" / f"{split}.csv").open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                relative = Path(row["video_path"].replace("\\", "/"))
                if relative.is_absolute() or ".." in relative.parts or relative.stem != row["video_name"]:
                    raise ValueError(f"Invalid video path: {relative}")
                source = root / "videos" / split / relative
                destination = output_root / split / relative.with_suffix("")
                if destination in seen:
                    raise ValueError(f"Duplicate video destination: {destination}")
                seen.add(destination)
                jobs.append((source, destination, count))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = pool.map(lambda job: extract_video(*job), jobs)
        counts = {"created": 0, "reused": 0}
        for done, result in enumerate(results, start=1):
            counts[result] += 1
            if done == 1 or done % 25 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} videos: {counts}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path,
                        help="Writable cache directory; default: ROOT/frames_native")
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val"])
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    try:
        prepare(args.root.resolve(), args.splits, args.num_frames, args.workers,
                args.output_root.resolve() if args.output_root is not None else None)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
