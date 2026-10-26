"""Two-stream score fusion of separately trained RGB and flow checkpoints.

As in Simonyan & Zisserman (2014), the video prediction is a weighted average
of the two streams' softmax scores, p = (1 - w) p_rgb + w p_flow. The weight w
is chosen on validation videos only (highest accuracy, then lowest NLL); the
equal-weight average w = 0.5 is reported alongside it. Test videos are scored
only with --test, after all model selection is finished.

python -m video_classification.fuse --rgb RGB/best.pt --flow FLOW/best.pt \\
    --output-dir outputs/v2/fusion [--test]
"""

import argparse
import json
from pathlib import Path

import torch

from .evaluate import load_checkpoint, metrics, predict

WEIGHTS = [round(0.05 * step, 2) for step in range(21)]


def _streams(rgb_path, flow_path, split, device, data_root):
    outputs = {}
    for name, path in (("rgb", rgb_path), ("flow", flow_path)):
        model, config, _ = load_checkpoint(path, data_root=data_root)
        if config["dataset"]["regime"] != "no_leakage":
            raise ValueError(f"{name} checkpoint was not trained on the corrected splits")
        outputs[name] = predict(model, config, split, device)
        outputs[name]["num_classes"] = config["dataset"]["num_classes"]
    rgb, flow = outputs["rgb"], outputs["flow"]
    if (rgb["video_ids"] != flow["video_ids"] or not torch.equal(rgb["labels"], flow["labels"])
            or rgb["class_to_label"] != flow["class_to_label"]):
        raise ValueError(f"RGB and flow {split} videos/labels are not aligned")
    return rgb, flow


def _combine(rgb, flow, weight):
    return (1 - weight) * rgb["probabilities"] + weight * flow["probabilities"]


def fuse(rgb_path, flow_path, output_dir, *, test=False, device="auto", data_root=None) -> dict:
    device = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                          else "cpu" if device == "auto" else device)
    rgb, flow = _streams(rgb_path, flow_path, "val", device, data_root)
    classes = rgb["num_classes"]
    sweep = []
    for weight in WEIGHTS:
        result = metrics(_combine(rgb, flow, weight), rgb["labels"], classes)
        sweep.append({"flow_weight": weight, "accuracy": result["accuracy"], "nll": result["nll"]})
    selected = max(sweep, key=lambda row: (row["accuracy"], -row["nll"]))["flow_weight"]
    report = {"rgb_checkpoint": str(Path(rgb_path).resolve()),
              "flow_checkpoint": str(Path(flow_path).resolve()),
              "selected_flow_weight": selected, "val_weight_sweep": sweep, "results": {}}
    splits = [("val", rgb, flow)]
    if test:
        splits.append(("test", *_streams(rgb_path, flow_path, "test", device, data_root)))
    for split, rgb_split, flow_split in splits:
        labels = rgb_split["labels"]
        report["results"][split] = {
            "rgb_only": metrics(rgb_split["probabilities"], labels, classes),
            "flow_only": metrics(flow_split["probabilities"], labels, classes),
            "fused_equal": metrics(_combine(rgb_split, flow_split, 0.5), labels, classes),
            f"fused_w{selected}": metrics(_combine(rgb_split, flow_split, selected), labels, classes),
        }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / ("fusion_val_test.json" if test else "fusion_val.json")
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Selected flow weight on validation: {selected}")
    for split, rows in report["results"].items():
        for name, row in rows.items():
            print(f"  {split:4s} {name:14s} accuracy {row['accuracy']:.1%} "
                  f"({row['correct']}/{row['videos']}); NLL {row['nll']:.4f}")
    print(f"Wrote {destination}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rgb", type=Path, required=True, help="RGB-stream checkpoint")
    parser.add_argument("--flow", type=Path, required=True, help="flow_stream checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test", action="store_true",
                        help="Also score the test split (final reporting only)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--data-root", help="Override the dataset root stored in both checkpoints")
    args = parser.parse_args()
    try:
        fuse(args.rgb, args.flow, args.output_dir, test=args.test, device=args.device,
             data_root=args.data_root)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
