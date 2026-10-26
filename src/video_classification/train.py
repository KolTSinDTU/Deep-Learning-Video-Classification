"""Train RGB, flow, and dual-stream models using video validation."""

import argparse
import copy
import csv
from datetime import datetime, timezone
from itertools import islice
import json
import math
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

from .config import MODEL_NAMES, PROJECT_ROOT, early_stopping_settings, load_config, validate_image_size
from .data import build_dataloaders
from .initialization import initialize_model
from .models import build_model


def _batches(loader, limit, description):
    count = len(loader) if limit is None else min(len(loader), limit)
    return tqdm(islice(loader, limit), total=count, desc=description,
                disable=not sys.stderr.isatty())


def _to_device(inputs, device):
    if isinstance(inputs, dict):
        return {key: value.to(device) for key, value in inputs.items()}
    return inputs.to(device)


def _forward(model, inputs):
    return model(**inputs) if isinstance(inputs, dict) else model(inputs)


def _train_epoch(model, loader, optimizer, device, limit=None, *, sample_unit="frame",
                 label_smoothing=0.0):
    model.train()
    loss_sum, correct, count = 0.0, 0, 0
    for inputs, labels in _batches(loader, limit, f"Training {sample_unit}s"):
        inputs, labels = _to_device(inputs, device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = _forward(model, inputs)
        loss = nn.functional.cross_entropy(logits, labels, label_smoothing=label_smoothing)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss; inspect inputs and learning rate")
        loss.backward()
        optimizer.step()
        count += labels.size(0)
        loss_sum += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
    return {"loss": loss_sum / count, "accuracy": correct / count, "count": count}


@torch.inference_mode()
def _validate(model, loader, device, limit=None, *, video_logits=False):
    """Evaluate video predictions, using stable cross-entropy for video logits."""
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    for videos, labels in _batches(loader, limit, "Validation videos"):
        videos, labels = _to_device(videos, device), labels.to(device)
        if video_logits:
            logits = _forward(model, videos)
            loss = nn.functional.cross_entropy(logits, labels, reduction="sum")
            predictions = logits.argmax(1)
        else:
            probabilities = model.predict_video(videos)
            # The per-frame model averages probabilities, so evaluate their NLL.
            log_probabilities = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
            loss = nn.functional.nll_loss(log_probabilities, labels, reduction="sum")
            predictions = probabilities.argmax(1)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite validation loss")
        count += labels.size(0)
        loss_sum += loss.item()
        correct += (predictions == labels).sum().item()
    return {"loss": loss_sum / count, "accuracy": correct / count, "count": count}


def build_optimizer(model, training: dict):
    """Adam (baseline), AdamW (decoupled weight decay), or SGD with momentum."""
    name = training.get("optimizer", "adam")
    rate, decay = training["learning_rate"], training.get("weight_decay", 0.0)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=rate, weight_decay=decay)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=rate, momentum=0.9, weight_decay=decay)
    # Exclude BN/bias parameters from weight decay, as is standard.
    decayed = [p for p in model.parameters() if p.ndim > 1]
    others = [p for p in model.parameters() if p.ndim <= 1]
    return torch.optim.AdamW([{"params": decayed, "weight_decay": decay},
                              {"params": others, "weight_decay": 0.0}], lr=rate)


def build_scheduler(optimizer, training: dict):
    """Per-epoch learning-rate multiplier: linear warmup, then cosine to zero."""
    if training.get("scheduler", "none") == "none":
        return None
    epochs = training["epochs"]
    warmup = min(training.get("warmup_epochs", 0), epochs - 1)

    def factor(epoch):  # epoch is the number of completed epochs
        if epoch < warmup:
            return (epoch + 1) / (warmup + 1)
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _augmentation_description(data: dict, flow_model: bool) -> str:
    if not data.get("augment", False):
        return "none"
    flip = "horizontal flip"
    if flow_model and data["flow_format"] == "rgb_png":
        flip += " (disabled for colour-coded flow)"
    return f"clip-consistent random resized crop (scale 0.5-1), {flip}, RGB colour jitter"


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def train(config: dict, *, smoke_test: bool = False) -> Path:
    """Train frames or videos and return a new timestamped run directory.

    Smoke tests use one epoch and at most two batches per split. They are saved
    separately and explicitly marked; their scores are not experiment results.
    Existing runs are never overwritten. This function does not load test data.
    Stop after patience epochs without a significant validation-loss decrease.
    Save both the highest-accuracy and lowest-loss validation checkpoints.
    """
    config = copy.deepcopy(config)
    model_name = config["model"]["name"]
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {model_name}")
    sample_unit = "frame" if model_name == "per_frame" else "video"
    data = config["dataset"]
    flow_model = model_name in {"dual_stream", "flow_stream"}
    validate_image_size(config["dataset"]["image_size"])
    training = config["training"]
    patience, min_delta = early_stopping_settings(training)
    training["early_stopping_patience"] = patience
    training["early_stopping_min_delta"] = min_delta
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
    mapping = loaders["train"].dataset.class_to_label
    initialization = initialize_model(model, config, mapping)
    optimizer = build_optimizer(model, training)
    scheduler = build_scheduler(optimizer, training)

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
        "optimizer": {"adam": "Adam", "adamw": "AdamW", "sgd": "SGD"}[training.get("optimizer", "adam")],
        "weight_decay": training.get("weight_decay", 0.0),
        "scheduler": training.get("scheduler", "none"),
        "warmup_epochs": training.get("warmup_epochs", 0),
        "label_smoothing": training.get("label_smoothing", 0.0),
        "backbone": config["model"].get("backbone", "small"),
        "augmentation": _augmentation_description(data, flow_model),
        "initialization": initialization,
        "frame_source": config["dataset"].get("frame_source", "provided"),
        "flow_format": data.get("flow_format") if flow_model else None,
        "flow_preprocessing": (
            (("RGB color PNG ToTensor, inverted so zero motion is 0"
              if data.get("input_normalization", "none") == "standardize"
              else "RGB color PNG ToTensor in [0, 1]") if data["flow_format"] == "rgb_png"
             else f"Pixel displacement vectors; spatial resize rescales u/v; divide by {data['flow_scale']}")
            if flow_model else None
        ),
        "preprocessing": (
            ("Native source resolution" if data["image_size"] == "native"
             else "Resize to image_size square") + "; RGB ToTensor in [0, 1]"
            + ("; ImageNet mean/std standardization"
               if data.get("input_normalization", "none") == "standardize" else "")
        ),
        "training_unit": sample_unit,
        "video_aggregation": {
            "per_frame": "Mean of frame softmax probabilities",
            "late_fusion": "Mean of frame features, then linear classification and softmax",
            "early_fusion": "Stack RGB frames in temporal order along channels before the CNN",
            "cnn3d": "Spatiotemporal 3D convolutions, then global pooling and classification",
            "dual_stream": ("Independent RGB/flow encoders; temporal mean RGB features; "
                            + ("stacked flow fields" if config["model"].get("flow_input", "mean") == "stack"
                               else "temporal mean flow features")
                            + "; concatenation and learned classifier"),
            "flow_stream": "Stack all flow fields along channels before the CNN (temporal stream)",
        }[model_name],
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    })

    print(f"Run directory: {run_dir}", flush=True)
    print(f"Model: {model_name}; device: {device}; "
          f"training: {len(loaders['train'].dataset)} {sample_unit}s; "
          f"validation: {len(loaders['val'].dataset)} videos", flush=True)
    if initialization["method"] != "random":
        print(f"Initialized weights from: {initialization['source_checkpoint']} "
              "(fresh optimizer)", flush=True)
    if smoke_test:
        print("SMOKE TEST: at most 2 batches per split; scores are for pipeline checks only.",
              flush=True)
    if patience:
        print(f"Early stopping: validation video loss; patience={patience}; "
              f"min_delta={min_delta:g}", flush=True)
    else:
        print("Early stopping: disabled", flush=True)

    train_loss_key = f"train_{sample_unit}_loss"
    train_accuracy_key = f"train_{sample_unit}_accuracy"
    train_count_key = f"train_{sample_unit}s"
    fields = ["epoch", train_loss_key, train_accuracy_key, train_count_key,
              "val_video_loss", "val_video_accuracy", "val_videos", "seconds",
              "early_stopping_bad_epochs", "learning_rate"]
    best_key = (-1.0, float("-inf"))
    best_epoch = 0
    minimum_loss = float("inf")
    best_loss_epoch = 0
    # Stopping uses the last significant improvement, while checkpoint saving
    # tracks every new minimum. Small decreases can accumulate past min_delta.
    reference_loss = float("inf")
    last_improvement_epoch = 0
    bad_epochs = 0
    started = time.perf_counter()
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, training["epochs"] + 1):
            epoch_started = time.perf_counter()
            print(f"Epoch {epoch}/{training['epochs']}", flush=True)
            current_rate = optimizer.param_groups[0]["lr"]
            training_metrics = _train_epoch(
                model, loaders["train"], optimizer, device, limit, sample_unit=sample_unit,
                label_smoothing=training.get("label_smoothing", 0.0),
            )
            if scheduler is not None:
                scheduler.step()
            validation_metrics = _validate(
                model, loaders["val"], device, limit, video_logits=(model_name != "per_frame")
            )
            if validation_metrics["loss"] < reference_loss - min_delta:
                reference_loss = validation_metrics["loss"]
                last_improvement_epoch = epoch
                bad_epochs = 0
            else:
                bad_epochs += 1
            stopped_early = bool(patience and bad_epochs >= patience
                                 and epoch < training["epochs"])
            stopping_state = {
                "enabled": patience > 0,
                "monitor": "val_video_loss",
                "patience": patience,
                "min_delta": min_delta,
                "reference_loss": reference_loss,
                "last_improvement_epoch": last_improvement_epoch,
                "bad_epochs": bad_epochs,
            }
            row = {
                "epoch": epoch,
                train_loss_key: training_metrics["loss"],
                train_accuracy_key: training_metrics["accuracy"],
                train_count_key: training_metrics["count"],
                "val_video_loss": validation_metrics["loss"],
                "val_video_accuracy": validation_metrics["accuracy"],
                "val_videos": validation_metrics["count"],
                "seconds": time.perf_counter() - epoch_started,
                "early_stopping_bad_epochs": bad_epochs,
                "learning_rate": current_rate,
            }
            # Prefer highest video accuracy, then lowest video NLL on ties.
            key = (validation_metrics["accuracy"], -validation_metrics["loss"])
            checkpoints_to_save = []
            if key > best_key:
                best_key, best_epoch = key, epoch
                checkpoints_to_save.append(("best.pt", "val_video_accuracy"))
            if validation_metrics["loss"] < minimum_loss:
                minimum_loss, best_loss_epoch = validation_metrics["loss"], epoch
                if training.get("save_best_loss_checkpoint", True):
                    checkpoints_to_save.append(("best_loss.pt", "val_video_loss"))
            if checkpoints_to_save:
                # Optimizer state roughly triples checkpoint size; it is only
                # needed to resume, which this trainer does not do.
                optimizer_state = (optimizer.state_dict()
                                   if training.get("save_optimizer_state", True) else None)
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer_state,
                    "config": config,
                    "class_to_label": mapping,
                    "epoch": epoch,
                    "validation": validation_metrics,
                    "smoke_test": smoke_test,
                    "initialization": initialization,
                    "early_stopping": stopping_state,
                }
                for filename, selection_metric in checkpoints_to_save:
                    destination = run_dir / filename
                    temporary = destination.with_suffix(".tmp")
                    torch.save({**checkpoint, "selection_metric": selection_metric}, temporary)
                    temporary.replace(destination)
            writer.writerow(row)
            stream.flush()
            _write_json(run_dir / "summary.json", {
                "smoke_test": smoke_test,
                "completed_epochs": epoch,
                "max_epochs": training["epochs"],
                "stopped_early": stopped_early,
                "stop_reason": ("early_stopping" if stopped_early else
                                "max_epochs" if epoch == training["epochs"] else "running"),
                "early_stopping": stopping_state,
                "best_epoch": best_epoch,
                "best_val_video_accuracy": best_key[0],
                "best_val_video_loss": -best_key[1],
                "checkpoint": str(run_dir / "best.pt"),
                "best_loss_epoch": best_loss_epoch,
                "minimum_val_video_loss": minimum_loss,
                "best_loss_checkpoint": (str(run_dir / "best_loss.pt")
                                         if training.get("save_best_loss_checkpoint", True) else None),
                "elapsed_seconds": time.perf_counter() - started,
            })
            print(f"  {sample_unit} loss {row[train_loss_key]:.4f}; "
                  f"validation video accuracy {row['val_video_accuracy']:.1%}; "
                  f"validation loss {row['val_video_loss']:.4f}; "
                  f"{row['seconds']:.1f}s", flush=True)
            if stopped_early:
                print(f"Early stopping at epoch {epoch}: validation video loss did not "
                      f"improve by more than {min_delta:g} for {patience} consecutive "
                      "epochs.", flush=True)
                break

    print(f"Best validation accuracy epoch: {best_epoch}; checkpoint: {run_dir / 'best.pt'}",
          flush=True)
    print(f"Lowest validation loss epoch: {best_loss_epoch}; loss: {minimum_loss:.4f}"
          + (f"; checkpoint: {run_dir / 'best_loss.pt'}"
             if training.get("save_best_loss_checkpoint", True) else ""), flush=True)
    return run_dir


def _positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def _nonnegative_float(value):
    number = float(value)
    if not 0 <= number < float("inf"):
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return number


def _image_size(value):
    if value == "native":
        return value
    return _positive_int(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epochs", type=_positive_int, help="Override the maximum number of epochs")
    parser.add_argument("--patience", type=_nonnegative_int,
                        help="Epochs without a significant validation-loss decrease; 0 disables stopping")
    parser.add_argument("--min-delta", type=_nonnegative_float,
                        help="Minimum absolute validation-loss decrease to reset patience")
    parser.add_argument("--image-size", type=_image_size,
                        help="native preserves source resolution; an integer resizes to a square")
    parser.add_argument("--batch-size", type=_positive_int,
                        help="Training batch size: frames for per_frame, videos for fusion/3D models")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=_nonnegative_int, help="Override experiment.seed")
    parser.add_argument("--num-workers", type=_nonnegative_int, help="Override training.num_workers")
    parser.add_argument("--output-dir", type=Path, help="Parent for new run directories")
    parser.add_argument("--init-checkpoint", type=Path,
                        help="Initialize late/early fusion, cnn3d (vgg_bn), or the dual-stream "
                             "RGB encoder from per-frame weights; starts a fresh optimizer")
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
            (args.patience, "training", "early_stopping_patience"),
            (args.min_delta, "training", "early_stopping_min_delta"),
            (args.seed, "experiment", "seed"),
            (args.num_workers, "training", "num_workers"),
        ):
            if argument is not None:
                config[section][key] = argument
        if args.init_checkpoint is not None:
            config["training"]["init_checkpoint"] = str(args.init_checkpoint)
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
