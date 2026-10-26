"""Generate the report table and combined loss plot from verified cluster runs."""

import csv
import hashlib
import json
import math
import os
from pathlib import Path

REPORT = Path(__file__).resolve().parent
PROJECT = REPORT.parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / "../../tmp/report-matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Explicit matching 50-epoch-budget runs; never silently pick by accuracy or recency.
CLUSTER = "hpc-results-20261021-100431/outputs"
RUNS = [
    ("Per-frame", "per_frame", f"{CLUSTER}/01_per_frame_seed42/20261021T072135_823976Z", 94538, "#0072B2"),
    ("Late fusion", "late_fusion", f"{CLUSTER}/02_late_fusion_seed42/20261021T073116_763728Z", 94538, "#D55E00"),
    ("Early fusion", "early_fusion", f"{CLUSTER}/03_early_fusion_seed42/20261021T074608_924815Z", 102314, "#009E73"),
    ("3D CNN", "cnn3d", f"{CLUSTER}/04_cnn3d_seed42/20261021T074842_549366Z", 280586, "#AA4499"),
]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    (REPORT / "figures").mkdir(exist_ok=True)
    (REPORT / "generated").mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7, 2.8), layout="constrained")
    rows, evidence, model_handles = [], [], []
    loss_max, last_epoch = 0.0, 0
    class_mapping = None
    for name, model_name, relative_run, parameters, color in RUNS:
        run = PROJECT / relative_run
        summary = read_json(run / "summary.json")
        config = read_json(run / "config.json")
        environment = read_json(run / "environment.json")
        mapping = read_json(run / "class_mapping.json")
        with (run / "history.csv").open(newline="", encoding="utf-8-sig") as stream:
            history = list(csv.DictReader(stream))
        assert class_mapping is None or mapping == class_mapping
        class_mapping = mapping
        assert history and not summary["smoke_test"]
        assert len(history) == summary["completed_epochs"]
        assert [int(r["epoch"]) for r in history] == list(range(1, len(history) + 1))
        assert config["model"]["name"] == model_name
        assert config["model"].get("normalization", "none") == "none"
        assert config["dataset"]["regime"] == "original"
        assert config["dataset"]["image_size"] == 224
        assert config["dataset"]["num_frames"] == 10
        assert config["experiment"]["seed"] == 42
        assert config["training"]["epochs"] == summary["max_epochs"] == 50
        assert config["training"]["learning_rate"] == 0.001
        assert config["training"]["early_stopping_patience"] == 5
        assert config["training"]["early_stopping_min_delta"] == 0.0001
        assert environment["initialization"]["method"] == "random"
        assert environment["optimizer"] == "Adam" and environment["device"] == "cuda"
        assert all(int(r["val_videos"]) == 120 for r in history)
        selected = min(history, key=lambda r: (-float(r["val_video_accuracy"]),
                                               float(r["val_video_loss"])))
        loss_selected = min(history, key=lambda r: float(r["val_video_loss"]))
        assert int(selected["epoch"]) == summary["best_epoch"]
        assert int(loss_selected["epoch"]) == summary["best_loss_epoch"]
        for key in ["accuracy", "loss"]:
            assert math.isclose(float(selected[f"val_video_{key}"]),
                                summary[f"best_val_video_{key}"])
        assert math.isclose(float(loss_selected["val_video_loss"]), summary["minimum_val_video_loss"])
        if summary["stopped_early"]:
            assert int(history[-1]["early_stopping_bad_epochs"]) == 5
            assert summary["stop_reason"] == "early_stopping"
        else:
            assert len(history) == 50 and summary["stop_reason"] == "max_epochs"

        unit = "frame" if model_name == "per_frame" else "video"
        train_key = f"train_{unit}_loss"
        examples = 5000 if unit == "frame" else 500
        assert all(int(r[f"train_{unit}s"]) == examples for r in history)
        epochs = [int(r["epoch"]) for r in history]
        train_losses = [float(r[train_key]) for r in history]
        val_losses = [float(r["val_video_loss"]) for r in history]
        assert all(math.isfinite(v) and v >= 0 for v in train_losses + val_losses)
        loss_max = max(loss_max, *train_losses, *val_losses)
        last_epoch = max(last_epoch, epochs[-1])
        ax.plot(epochs, train_losses, color=color, linewidth=1.4, linestyle="-", alpha=0.9)
        ax.plot(epochs, val_losses, color=color, linewidth=1.7, linestyle=(0, (4, 2)))
        model_handles.append(Patch(facecolor=color, edgecolor="none", label=name))
        rows.append(f"{name} & {summary['best_epoch']}/{len(history)} & "
                    f"{100 * summary['best_val_video_accuracy']:.1f} & "
                    f"{summary['best_val_video_loss']:.3f} " + r"\\")
        updates = math.ceil(examples / config["training"]["batch_size"])
        evidence.append({
            "model": name, "run": relative_run, "parameter_count": parameters,
            "status": "completed cluster training; original-split validation only",
            "selected_epoch": summary["best_epoch"], "selection": "max validation video accuracy; min loss breaks ties",
            "validation_accuracy": summary["best_val_video_accuracy"],
            "validation_loss_at_selected_epoch": summary["best_val_video_loss"],
            "minimum_validation_loss": summary["minimum_val_video_loss"],
            "minimum_loss_epoch": summary["best_loss_epoch"],
            "accuracy_at_minimum_loss_epoch": float(loss_selected["val_video_accuracy"]),
            "completed_epochs": len(history), "max_epochs": summary["max_epochs"],
            "stop_reason": summary["stop_reason"], "elapsed_seconds": summary["elapsed_seconds"],
            "batch_size": config["training"]["batch_size"], "updates_per_epoch": updates,
            "total_optimizer_updates": updates * len(history),
            "training_loss_column": train_key, "validation_loss_column": "val_video_loss",
            "final_training_loss": train_losses[-1], "final_validation_loss": val_losses[-1],
            "final_training_accuracy": float(history[-1][f"train_{unit}_accuracy"]),
            "image_size": 224, "seed": 42, "device": environment["device_name"],
            "source_sha256": {filename: hashlib.sha256((run / filename).read_bytes()).hexdigest()
                              for filename in ["config.json", "summary.json", "history.csv", "environment.json"]},
        })
    split_handles = [Line2D([0], [0], color="#444444", linewidth=1.5, label="Training"),
                     Line2D([0], [0], color="#444444", linewidth=1.7,
                            linestyle=(0, (4, 2)), label="Validation")]
    ax.set(xlabel="Epoch", ylabel="Loss", xlim=(1, last_epoch),
           ylim=(0, math.ceil(loss_max * 2) / 2), xticks=[1, 10, 20, 30, 40, 50])
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.legend(handles=model_handles + split_handles, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), ncol=6, frameon=False, fontsize=8,
              handlelength=1.6, columnspacing=1.15, borderaxespad=0)
    fig.savefig(REPORT / "figures/train_val_loss.pdf")
    fig.savefig(REPORT / "figures/train_val_loss.png", dpi=200)
    plt.close(fig)
    table = r"""% Generated by build_assets.py from explicit completed cluster runs.
\begin{table}[t]
    \centering
    \caption{Original-split validation results from the cluster runs.
    Epochs denotes selected/completed epochs. Accuracy and loss come
    from the same accuracy-selected checkpoint, not the final epoch.
    No test results are included.}
    \label{tab:baselines}
    \begin{tabular}{lrrr}
        \toprule
        Model & Epochs & Val. (\%) & Val. loss \\
        \midrule
""" + "\n".join("        " + row for row in rows) + r"""
        \bottomrule
    \end{tabular}
\end{table}
"""
    (REPORT / "generated/baseline_table.tex").write_text(table, encoding="utf-8")
    (REPORT / "generated/evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    for item in evidence:
        print(f"{item['model']}: {item['validation_accuracy']:.4f} validation accuracy; "
              f"epoch {item['selected_epoch']}/{item['completed_epochs']}; "
              f"{item['total_optimizer_updates']} optimizer updates")
    print("Generated a single-panel plot with all eight train/validation curves.")


if __name__ == "__main__":
    main()
