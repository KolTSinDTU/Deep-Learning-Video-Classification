"""Evaluate a saved checkpoint on whole videos; reserve test for final reporting.

python -m video_classification.evaluate --checkpoint RUN/best.pt --split val
"""

import argparse
import io
import json
from pathlib import Path

import torch

from .data import build_eval_loader
from .models import build_model


def load_checkpoint(path: str | Path, *, data_root: str | None = None):
    """Rebuild the trained model in eval mode; return (model, config, checkpoint)."""
    path = Path(path).expanduser().resolve()
    checkpoint = torch.load(io.BytesIO(path.read_bytes()), map_location="cpu", weights_only=True)
    if checkpoint.get("smoke_test"):
        raise ValueError(f"{path} is a smoke-test checkpoint; its scores are not results")
    config = checkpoint["config"]
    if data_root is not None:
        config["dataset"]["root"] = str(Path(data_root).expanduser().resolve())
    # Weights come from the checkpoint; never download pretrained weights here.
    config["model"]["pretrained"] = False
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.eval(), config, checkpoint


@torch.inference_mode()
def predict(model, config: dict, split: str, device: torch.device) -> dict:
    """Return ordered video ids, labels, and video probabilities [N, C]."""
    loader = build_eval_loader(config, split)
    model.to(device)
    probabilities, labels = [], []
    for inputs, batch_labels in loader:
        if isinstance(inputs, dict):
            output = model.predict_video(**{k: v.to(device) for k, v in inputs.items()})
        else:
            output = model.predict_video(inputs.to(device))
        probabilities.append(output.float().cpu())
        labels.append(batch_labels)
    return {"video_ids": [record[0] for record in loader.dataset.records],
            "labels": torch.cat(labels), "probabilities": torch.cat(probabilities),
            "class_to_label": loader.dataset.class_to_label}


def metrics(probabilities: torch.Tensor, labels: torch.Tensor, num_classes: int) -> dict:
    """Video accuracy, NLL, macro F1, and a confusion matrix (rows = true class)."""
    predictions = probabilities.argmax(1)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for true, predicted in zip(labels.tolist(), predictions.tolist()):
        confusion[true, predicted] += 1
    true_positive = confusion.diag().float()
    precision = true_positive / confusion.sum(0).clamp_min(1)
    recall = true_positive / confusion.sum(1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    nll = -probabilities.clamp_min(1e-12).log()[torch.arange(len(labels)), labels].mean()
    return {"videos": len(labels), "correct": int((predictions == labels).sum()),
            "accuracy": float((predictions == labels).float().mean()), "nll": float(nll),
            "macro_f1": float(f1.mean()), "confusion_matrix": confusion.tolist()}


def evaluate(checkpoint: str, split: str = "val", *, device: str = "auto",
             data_root: str | None = None) -> dict:
    """Evaluate one checkpoint and write eval_<split>.json next to it."""
    if split not in {"val", "test"}:
        raise ValueError("split must be val or test")
    device = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                          else "cpu" if device == "auto" else device)
    model, config, saved = load_checkpoint(checkpoint, data_root=data_root)
    result = predict(model, config, split, device)
    if result["class_to_label"] != saved["class_to_label"]:
        raise ValueError("Evaluation class mapping disagrees with the checkpoint")
    report = {
        "checkpoint": str(Path(checkpoint).resolve()), "split": split,
        "model": config["model"]["name"], "regime": config["dataset"]["regime"],
        "epoch": saved["epoch"],
        **metrics(result["probabilities"], result["labels"], config["dataset"]["num_classes"]),
        "class_to_label": result["class_to_label"],
        "predictions": [{"video": video, "label": int(label), "probabilities": row.tolist()}
                        for video, label, row in zip(result["video_ids"], result["labels"],
                                                     result["probabilities"])],
    }
    destination = Path(checkpoint).resolve().parent / f"eval_{split}.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{report['model']} ({report['regime']}) {split}: accuracy {report['accuracy']:.1%} "
          f"({report['correct']}/{report['videos']}); NLL {report['nll']:.4f}; "
          f"macro F1 {report['macro_f1']:.3f} -> {destination}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--data-root", help="Override the dataset root stored in the checkpoint")
    args = parser.parse_args()
    try:
        evaluate(args.checkpoint, args.split, device=args.device, data_root=args.data_root)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
