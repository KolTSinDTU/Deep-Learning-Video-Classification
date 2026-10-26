"""Plot v2 training/validation curves for the report.

python report/plot_v2_curves.py hpc-results-20261022-131708/outputs [accuracy|loss]

Writes report/figures/v2_<metric>_{original,corrected}.pdf (and .png). Only
runs that completed all epochs are used; the newest one per experiment is
taken.
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Categorical slots 1-5 of the validated dataviz palette, in fixed order.
MODELS = [("per_frame", "Per-frame", "#2a78d6"), ("late_fusion", "Late fusion", "#eb6834"),
          ("early_fusion", "Early fusion", "#1baf7a"), ("cnn3d", "3D CNN", "#eda100"),
          ("flow_stream", "Flow stream", "#e87ba4")]
PANELS = [("Original (leaky) splits", "original",
           {"per_frame": "01_per_frame", "late_fusion": "02_late_fusion",
            "early_fusion": "03_early_fusion", "cnn3d": "04_cnn3d"}),
          ("Corrected splits", "corrected",
           {"per_frame": "05_per_frame_noleak", "late_fusion": "06_late_fusion_noleak",
            "early_fusion": "09_early_fusion_noleak", "cnn3d": "10_cnn3d_noleak",
            "flow_stream": "07_flow_stream"})]
LABELS = {"accuracy": "Video accuracy (%)", "loss": "Cross-entropy loss"}


def history(outputs: Path, experiment: str, metric: str):
    runs = [run for run in sorted((outputs / experiment / "seed42").glob("2*"))
            if json.loads((run / "summary.json").read_text())["stop_reason"] == "max_epochs"]
    rows = list(csv.DictReader((runs[-1] / "history.csv").open()))
    train_key = next(k for k in rows[0] if k.startswith("train_") and k.endswith(metric))
    scale = 100 if metric == "accuracy" else 1
    epochs = [int(r["epoch"]) for r in rows]
    return (epochs, [scale * float(r[train_key]) for r in rows],
            [scale * float(r[f"val_video_{metric}"]) for r in rows])


def plot_panel(outputs: Path, title: str, experiments: dict, metric: str, destination: Path):
    """One self-contained plot per split, sized for one CVPR column."""
    figure, axis = plt.subplots(figsize=(3.35, 2.15))
    handles = []
    for key, label, color in MODELS:
        if key not in experiments:
            continue
        epochs, train, val = history(outputs, experiments[key], metric)
        axis.plot(epochs, train, color=color, linewidth=1.0, linestyle=(0, (3, 1.5)))
        axis.plot(epochs, val, color=color, linewidth=1.5)
        handles.append(Line2D([], [], color=color, linewidth=2.0, label=label))
    handles += [Line2D([], [], color="#52514e", linewidth=1.5, label="Validation"),
                Line2D([], [], color="#52514e", linewidth=1.0, linestyle=(0, (3, 1.5)),
                       label="Training")]
    axis.set_title(title, fontsize=9, pad=4)
    axis.set_xlim(1, 60)
    if metric == "accuracy":
        axis.set_ylim(0, 102)
    else:
        axis.set_ylim(bottom=0)
    axis.set_xticks([1, 10, 20, 30, 40, 50, 60])
    axis.set_xlabel("Epoch")
    axis.set_ylabel(LABELS[metric])
    axis.grid(axis="y", linewidth=0.4, color="#d6d6d3")
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.tick_params(length=2.5, pad=2)
    figure.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7,
                  handlelength=1.8, columnspacing=0.9, handletextpad=0.4, borderaxespad=0.2)
    figure.subplots_adjust(left=0.13, right=0.97, top=0.90, bottom=0.33)
    figure.savefig(destination)
    figure.savefig(destination.with_suffix(".png"), dpi=300)
    plt.close(figure)
    print(f"Wrote {destination}")


def main():
    outputs = Path(sys.argv[1])
    metric = sys.argv[2] if len(sys.argv) > 2 else "accuracy"
    if metric not in LABELS:
        raise SystemExit("metric must be accuracy or loss")
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 8, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.6, "pdf.fonttype": 42})
    figures = Path(__file__).resolve().parent / "figures"
    for title, name, experiments in PANELS:
        plot_panel(outputs, title, experiments, metric, figures / f"v2_{metric}_{name}.pdf")


if __name__ == "__main__":
    main()
