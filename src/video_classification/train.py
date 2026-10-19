"""Train the per-frame baseline and select checkpoints using validation videos."""

import argparse
import copy
import csv
from datetime import datetime, timezone
from itertools import islice
import json
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import torch
from torch import nn
import torchvision
from tqdm import tqdm

from .config import PROJECT_ROOT, load_config
from .data import build_dataloaders
from .models import build_model


def _batches(loader, limit, description):
    count = len(loader) if limit is None else min(len(loader), limit)
    return tqdm(islice(loader, limit), total=count, desc=description,
                disable=not sys.stderr.isatty())


def _train_epoch(model, loader, optimizer, device, limit=None):
    model.train()
    loss_sum, correct, count = 0.0, 0, 0
    for frames, labels in _batches(loader, limit, "Training frames"):
        frames, labels = frames.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(frames)
        loss = nn.functional.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss; inspect inputs and learning rate")
        loss.backward()
        optimizer.step()
        count += labels.size(0)
        loss_sum += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
    return {"loss": loss_sum / count, "accuracy": correct / count, "count": count}


@torch.inference_mode()
def _validate(model, loader, device, limit=None):
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    for videos, labels in _batches(loader, limit, "Validation videos"):
        videos, labels = videos.to(device), labels.to(device)
        probabilities = model.predict_video(videos)
        # These are averaged probabilities, so use NLL rather than cross-entropy.
        log_probabilities = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
        loss = nn.functional.nll_loss(log_probabilities, labels, reduction="sum")
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite validation loss")
        count += labels.size(0)
        loss_sum += loss.item()
        correct += (probabilities.argmax(1) == labels).sum().item()
    return {"loss": loss_sum / count, "accuracy": correct / count, "count": count}


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def train(config: dict, *, smoke_test: bool = False) -> Path:
    """Train frames, validate videos, and return a new timestamped run directory.

    Smoke tests use one epoch and at most two batches per split. They are saved
    separately and explicitly marked; their scores are not experiment results.
    Existing runs are never overwritten. This function does not load test data.
    """
    config = copy.deepcopy(config)
    if config["model"]["name"] != "per_frame":
        raise NotImplementedError("Training currently supports the per_frame model")
    if config["dataset"]["image_size"] < 8:
        raise ValueError("dataset.image_size must be at least 8 for PerFrameCNN")
    training = config["training"]
    requested_device = training.get("device", "auto")
    if requested_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("training.device must be auto, cpu, or cuda")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu or --device auto")
    device = torch.device(
        "cuda" if requested_device == "auto" and torch.cuda.is_available()
        else "cpu" if requested_device == "auto" else requested_device
    )
    training["device"] = str(device)
    training["smoke_test"] = smoke_test
    if smoke_test:
        training["epochs"] = 1
    limit = 2 if smoke_test else None

    seed = config["experiment"]["seed"]
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    loaders = build_dataloaders(config)
    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"])
    mapping = loaders["train"].dataset.class_to_label

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = Path(config["experiment"]["output_dir"]) / (
        f"smoke_{stamp}" if smoke_test else stamp
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    config["experiment"]["output_dir"] = str(run_dir)
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "class_mapping.json", mapping)
    _write_json(run_dir / "environment.json", {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "numpy": np.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "torch_threads": torch.get_num_threads(),
        "optimizer": "Adam",
        "preprocessing": "Resize to image_size square; RGB ToTensor in [0, 1]",
        "video_aggregation": "Mean of frame softmax probabilities",
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    })

    print(f"Run directory: {run_dir}", flush=True)
    print(f"Device: {device}; training: {len(loaders['train'].dataset)} frames; "
          f"validation: {len(loaders['val'].dataset)} videos", flush=True)
    if smoke_test:
        print("SMOKE TEST: at most 2 batches per split; scores are for pipeline checks only.",
              flush=True)

    fields = ["epoch", "train_frame_loss", "train_frame_accuracy", "train_frames",
              "val_video_loss", "val_video_accuracy", "val_videos", "seconds"]
    best_key = (-1.0, float("-inf"))
    best_epoch = 0
    started = time.perf_counter()
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, training["epochs"] + 1):
            epoch_started = time.perf_counter()
            print(f"Epoch {epoch}/{training['epochs']}", flush=True)
            training_metrics = _train_epoch(model, loaders["train"], optimizer, device, limit)
            validation_metrics = _validate(model, loaders["val"], device, limit)
            row = {
                "epoch": epoch,
                "train_frame_loss": training_metrics["loss"],
                "train_frame_accuracy": training_metrics["accuracy"],
                "train_frames": training_metrics["count"],
                "val_video_loss": validation_metrics["loss"],
                "val_video_accuracy": validation_metrics["accuracy"],
                "val_videos": validation_metrics["count"],
                "seconds": time.perf_counter() - epoch_started,
            }
            # Prefer highest video accuracy, then lowest video NLL on ties.
            key = (validation_metrics["accuracy"], -validation_metrics["loss"])
            if key > best_key:
                best_key, best_epoch = key, epoch
                temporary = run_dir / "best.tmp"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "class_to_label": mapping,
                    "epoch": epoch,
                    "validation": validation_metrics,
                    "smoke_test": smoke_test,
                }, temporary)
                temporary.replace(run_dir / "best.pt")
            writer.writerow(row)
            stream.flush()
            _write_json(run_dir / "summary.json", {
                "smoke_test": smoke_test,
                "completed_epochs": epoch,
                "best_epoch": best_epoch,
                "best_val_video_accuracy": best_key[0],
                "best_val_video_loss": -best_key[1],
                "checkpoint": str(run_dir / "best.pt"),
                "elapsed_seconds": time.perf_counter() - started,
            })
            print(f"  frame loss {row['train_frame_loss']:.4f}; "
                  f"validation video accuracy {row['val_video_accuracy']:.1%}; "
                  f"validation loss {row['val_video_loss']:.4f}; "
                  f"{row['seconds']:.1f}s", flush=True)

    print(f"Best validation epoch: {best_epoch}; checkpoint: {run_dir / 'best.pt'}", flush=True)
    return run_dir


def _positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epochs", type=_positive_int, help="Override the number of epochs")
    parser.add_argument("--image-size", type=_positive_int, help="Override the square frame size")
    parser.add_argument("--batch-size", type=_positive_int, help="Override the training frame batch size")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--output-dir", type=Path, help="Parent for new run directories")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Check 2 train/val batches in one epoch; save separately")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        for argument, section, key in (
            (args.epochs, "training", "epochs"),
            (args.image_size, "dataset", "image_size"),
            (args.batch_size, "training", "batch_size"),
            (args.device, "training", "device"),
        ):
            if argument is not None:
                config[section][key] = argument
        if args.output_dir is not None:
            output = args.output_dir.expanduser()
            if not output.is_absolute():
                output = PROJECT_ROOT / output
            config["experiment"]["output_dir"] = str(output.resolve())
        train(config, smoke_test=args.smoke_test)
    except (OSError, ValueError, NotImplementedError) as error:
        parser.error(str(error))
    except KeyboardInterrupt:
        parser.exit(130, "\nStopped. Outputs from completed epochs remain in the printed run directory.\n")


if __name__ == "__main__":
    main()
