# Video classification — 2026

Project scaffold for `Project_2_VideoClassification.pdf`. The assignment PDF is
the source of requirements. The 2025 project has not been copied or modified.

## Status

The four single-stream training pipelines are ready: per-frame, late fusion,
early fusion, and 3D CNN. These include the frame/video data adapter, training
loop, validation, and checkpoint saving, alongside experiment configurations,
a configuration checker, notebook, and report outline. Dual stream and standalone
final test evaluation remain **implementation tasks**, with interfaces and TODOs
in `src/`. Datasets and Python environments are local dependencies excluded from
Git. Limited smoke tests check all four pipelines; experiment results belong
under `outputs/`.

## Layout

```text
configs/                     One TOML file per experiment
data/                        Local datasets and course-provided datasets.py
notebooks/01_explore_data.ipynb
outputs/                     Checkpoints, metrics, predictions, and figures
report/outline.md            Report plan and requirements checklist
src/video_classification/
    cli.py                   Validate and display an experiment
    config.py                Configuration loading and validation
    data.py                  Course CSV/frame adapter for frame or video training
    models/                  Four single-stream approaches and dual stream
    train.py                 2D/3D model training and video validation
    evaluate.py              Evaluation implementation starting point
```

## Environment setup

Use Python 3.11–3.13 (3.12 is a suitable starting point). From this directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[notebooks]"
.\.venv\Scripts\python.exe -m video_classification.cli --config configs/01_per_frame.toml
```

On Linux or an allocated HPC compute node:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[notebooks]'
.venv/bin/python -m video_classification.cli --config configs/01_per_frame.toml
```

The PyTorch/torchvision pair is pinned together. Choose the appropriate PyTorch
wheel source for your machine if you need CUDA; GPU availability depends on the
installed build and driver. No cluster queue or module names are assumed here.

The configuration checker only needs Python's standard library. Before installing
the package you can run it from this folder with:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m video_classification.cli --config configs/01_per_frame.toml
```

## Add the course data

1. Download the designated dataset and **2026 course version** of `datasets.py`
   from DTU Learn. Follow `data/README.md` for their locations.
2. Inspect the actual CSV headers, image layout, label mapping, and loader API in
   the exploration notebook. CSV names and schemas are deliberately not guessed.
3. Set `dataset.root` in each configuration. Relative paths resolve against this
   project directory, independently of the shell's working directory. For HPC,
   the brief identifies `/dtu/datasets1/02516/ucf101noleakage`; confirm the actual
   directory name on the cluster.
4. Check the root exists with `video-project --config configs/01_per_frame.toml
   --require-data` (put the command on one line). This checks the directory only;
   dataset integrity and split checks belong in the loader implementation.

## Experiment plan

| Config | Assignment task | Intended approach |
| --- | --- | --- |
| `01_per_frame.toml` | 4.1 | Train on frames; aggregate frame predictions per video |
| `02_late_fusion.toml` | 4.1 | Shared 2D encoder; pool frame features before classification |
| `03_early_fusion.toml` | 4.1 | Stack RGB frames into channels before the 2D CNN |
| `04_cnn3d.toml` | 4.1 | 3D CNN on `[B, C, T, H, W]` clips |
| `05_no_leakage.toml` | 4.2.1 | Retrain the selected 4.1 model on the corrected splits |
| `06_dual_stream.toml` | 4.2.2 | Combine RGB and optical-flow streams on corrected splits |

Hyperparameters are starting values, not tuned results. `05_no_leakage.toml`
initially names `per_frame`; **replace it with the validation-selected model and matching hyperparameters** from 4.1 before the comparison. Never choose using test
accuracy. The corrected split sizes are not assumed to match the original data.

The original dataset has 10 classes, 10 sampled RGB frames per video, and
500/120/120 train/validation/test videos. Do not infer the numeric class mapping
from the order of names in the brief; read it from the provided CSVs.

## Implementation order

1. Run the four single-stream baselines below. The adapter checks frame availability,
   class mappings, and exact train/validation video overlap. Subject identity
   still needs a separate audit; these checks do not fix the original leakage.
2. Implement standalone final evaluation for the trained models.
3. Record seeds, configurations, validation curves, best checkpoints, and metrics
   under separate `outputs/<experiment>/` directories. Use validation for model
   selection and the test split only for final evaluation.
4. Re-run the selected model on the corrected splits, then implement dual stream.
   Inspect the flow representation before selecting channels, normalization, or
   augmentations; `flows-png` may be visualizations rather than raw vector data.
5. Write the report with the **course-provided CVPR template** using the outline.

Launch the notebook after installation with `.venv/Scripts/jupyter-lab.exe` on
Windows or `.venv/bin/jupyter-lab` on Linux. Select the project virtual environment
as the kernel.

## Per-frame baseline

`models/per_frame.py` implements `PerFrameCNN`, a small CNN trained from scratch:
three convolution/ReLU/max-pooling blocks (32, 64, 128 channels), global average
pooling, and a linear classifier. There are no pretrained weights or downloads.
Use floating-point RGB inputs, initially scaled to `[0, 1]`, with the same
preprocessing at training and evaluation. The configured image size is 224.

- `model(frames)` takes `[B, 3, H, W]` and returns **logits** `[B, num_classes]`.
  Train with cross-entropy and each frame's video label.
- `model.predict_video(videos)` takes `[B, 3, T, H, W]` and returns
  **probabilities** `[B, num_classes]`. It applies softmax separately to each
  frame, then averages probabilities over time. Frame order does not affect this
  baseline. Compute validation/test accuracy using one prediction per video.
- Video inference processes all `B * T` frames together. Reduce the video batch
  size if memory becomes limiting. Call `eval()` and use `inference_mode()` when
  evaluating; `predict_video` leaves these settings under caller control.

From the project directory, this checks the model on synthetic input:

```python
import torch
from video_classification.config import load_config
from video_classification.models import build_model

config = load_config("configs/01_per_frame.toml")
model = build_model(config)
frames = torch.rand(2, 3, 224, 224)
labels = torch.tensor([0, 1])
loss = torch.nn.functional.cross_entropy(model(frames), labels)
loss.backward()

model.eval()
with torch.inference_mode():
    videos = torch.rand(1, 3, 10, 224, 224)
    probabilities = model.predict_video(videos)  # [1, 10]
    predictions = probabilities.argmax(dim=1)  # one label per video
```

Run the frame-training and video-aggregation checks with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Train the per-frame baseline

Run these commands from the `2026/Video Classification` directory. The local
configuration points to `data/ucf101/ufc10`, where the extracted frames and
metadata are stored. The adapter follows the provided `src/datasets.py` layout
using portable paths; it does not require changing the supplied reference file.

For a quick pipeline check (two batches per split, one epoch):

```powershell
.\.venv\Scripts\python.exe -m video_classification.train --config configs/01_per_frame.toml --smoke-test --image-size 112
```

For a first training experiment on the laptop (two full epochs at 112 pixels):

```powershell
.\.venv\Scripts\python.exe -m video_classification.train --config configs/01_per_frame.toml --epochs 2 --image-size 112
```

For the configured baseline (up to 20 epochs at 224 pixels):

```powershell
.\.venv\Scripts\python.exe -m video_classification.train --config configs/01_per_frame.toml
```

The command uses Adam with the configured learning rate, shuffles training
frames, and validates using averaged frame probabilities for each video.
Validation loss is the negative log-likelihood of those averaged probabilities.
`best.pt` has the highest validation video accuracy, with lower validation
loss breaking ties. `best_loss.pt` has the lowest validation video loss.
Training stops early when that loss stops improving significantly, as described
below. Test data is not loaded during training.
Resizing and scaling to `[0, 1]` are identical for training and validation;
this initial baseline does not use augmentation or pretrained weights.

Each invocation creates a new timestamped folder under
`outputs/01_per_frame_seed42/` and prints its full path. It saves `best.pt`, `best_loss.pt`,
`history.csv`, the effective `config.json`, `class_mapping.json`, environment
details, and a summary. Smoke folders start with `smoke_`, and their files are
marked accordingly; their partial-split scores are not experiment results.

You can override `--epochs`, `--image-size`, `--batch-size`, `--device`, and
`--output-dir` without editing the TOML file. `--device auto` uses CUDA when
available and otherwise CPU. Validation defaults to one video per batch to
keep memory use low; set `training.validation_batch_size` in TOML to change it.
Press Ctrl+C to stop. Completed epoch logs and the best saved checkpoint remain;
rerunning starts a fresh run, not a resumed one.

## Early stopping on validation loss

All implemented training modes monitor the mean validation **video loss** after
each epoch. The per-frame model uses NLL of averaged frame probabilities; the
fusion and 3D models use cross-entropy of video logits. The default settings,
also included in the supplied TOML configurations, are:

```toml
[training]
epochs = 20                       # Maximum number of epochs
early_stopping_patience = 5
early_stopping_min_delta = 0.0001
```

The first epoch establishes a reference loss. A decrease **greater than**
`min_delta` from the last significant improvement resets the patience counter.
Otherwise the counter increases. Five consecutive such epochs stop training.
Ties do not reset patience; small decreases can accumulate until they exceed
the threshold. Accuracy changes do not affect the stopping decision.

Override these settings from the command line:

```powershell
.\.venv\Scripts\python.exe -m video_classification.train --config configs/04_cnn3d.toml --epochs 50 --patience 5 --min-delta 0.0001
```

`--patience 0` disables early stopping for a fixed-epoch experiment.
`--smoke-test` still runs at most one epoch with two batches per split.

- `best_loss.pt` saves the lowest observed validation loss, including a new
  minimum too small to reset patience. Use this checkpoint for loss-based
  model selection; it may be from an epoch before training stopped.
- `best.pt` retains the existing accuracy-based selection rule.
- `history.csv` records `early_stopping_bad_epochs`.
- `summary.json` records the completed/max epoch counts, stop reason,
  stopping settings and counter, and paths/epochs for both checkpoints.
  `minimum_val_video_loss` belongs to `best_loss.pt`;
  `best_val_video_loss` remains the loss at the best-accuracy checkpoint.

The trainer returns the run directory, not an in-memory model. Reload the
chosen checkpoint for evaluation. Earlier runs are preserved.

## Late fusion with the same CNN

`models/late_fusion.py` implements `LateFusionCNN`. It reuses the encoder and
linear classifier architecture from `PerFrameCNN`, with fresh weights for each
experiment. All weights are trained together; no pretrained weights or previous
per-frame checkpoint are loaded automatically. Both models have the same
parameter count.

The model takes `[B, 3, T, H, W]`, encodes the `B*T` frames using the shared CNN,
averages their 128-dimensional features within each video, and applies the
classifier once per video. `forward()` returns video logits;
`predict_video()` applies softmax to those logits. Mean fusion ignores frame
order. With this linear classifier, feature averaging is equivalent to averaging
frame logits; the per-frame baseline instead averages frame probabilities.

Training and validation both use complete clips and one target per video.
Cross-entropy is computed directly from the video logits for numerical stability.
The `batch_size` setting counts **videos**, with a default of 2 (20 frames for
the course data), while validation uses 1 video per batch. Resize and `[0, 1]`
scaling are the same as in the per-frame baseline. Both configs default to 224
pixels; use matching resolutions when comparing results, and record batch sizes.

From the project directory, check the pipeline on limited batches:

```powershell
.\.venv\Scripts\python.exe -m video_classification.train --config configs/02_late_fusion.toml --smoke-test
```

For a first experiment with two full epochs at 112 pixels:

```powershell
.\.venv\Scripts\python.exe -m video_classification.train --config configs/02_late_fusion.toml --epochs 2 --image-size 112
```

Omit `--epochs 2 --image-size 112` to use up to 20 epochs at 224 pixels.
Runs go into new timestamped folders under `outputs/02_late_fusion_seed42/`.
The history records `train_video_loss`, `train_video_accuracy`, and
`train_videos`; per-frame runs retain their existing frame-level training columns.
Both modes select the best checkpoint using validation video accuracy, with
lower validation loss breaking ties. Test data is not loaded during training.

## Early fusion by stacking RGB frames

`models/early_fusion.py` implements `EarlyFusionCNN`. It uses the same CNN
architecture as the per-frame model, widening its first convolution to accept
`3 * dataset.num_frames` channels. All parameters start from fresh random
initialization and are trained together; no pretrained weights are used.

Input clips `[B, 3, T, H, W]` are rearranged to `[B, 3*T, H, W]` with channels
ordered `R1, G1, B1, R2, G2, B2, ...`. For the course's 10 frames, the input has
30 channels. The first convolution combines frames before any feature pooling.
Frame order can affect predictions, and the model requires exactly the number
of frames configured when it was created. This is a 2D CNN: time is represented
by input channels, not a separate dimension for temporal convolutions.

`forward()` returns one set of logits per video; `predict_video()` applies
softmax to those logits. Training and validation use video cross-entropy, with
the same preprocessing, checkpoint selection, and metrics as late fusion.
The default batch size is 2 videos, with 1 video per validation batch.

Check limited batches at the default resolution:

```powershell
.\.venv\Scripts\python.exe -m video_classification.train --config configs/03_early_fusion.toml --smoke-test
```

Start with two full epochs at 112 pixels:

```powershell
.\.venv\Scripts\python.exe -m video_classification.train --config configs/03_early_fusion.toml --epochs 2 --image-size 112
```

Omit `--epochs 2 --image-size 112` for up to 20 epochs at 224 pixels.
Outputs go to new timestamped folders under `outputs/03_early_fusion_seed42/`.
Use matching resolutions and frame sampling when comparing models.

## 3D CNN across space and time

`models/cnn3d.py` implements `CNN3D`, trained from scratch. It takes
`[B, 3, T, H, W]` directly and keeps time as a separate dimension. Three
`3 × 3 × 3` convolution/ReLU blocks produce 32, 64, and 128 feature channels.
Each kernel combines neighboring frames and spatial pixels, using the same
weights as it slides over time. This allows predictions to depend on frame order.

Each block uses max pooling with size `(1, 2, 2)`, reducing height and width
while preserving the time axis. Global average pooling then combines time,
height, and width into 128 features, and a linear classifier produces one set
of logits per video. The model supports any positive clip length; the dataset
configuration supplies 10 frames per video. Height and width must be at least 8.

`forward()` returns video logits; `predict_video()` applies softmax.
Training and validation use video cross-entropy, the same RGB `[0, 1]`
preprocessing, and the same checkpoint selection as the fusion models.
Both training and validation default to **1 video per batch** to limit memory
use on the laptop. No pretrained weights are used.

Check limited batches at the default 224-pixel resolution:

```powershell
.\.venv\Scripts\python.exe -m video_classification.train --config configs/04_cnn3d.toml --smoke-test
```

Start with two full epochs at 112 pixels:

```powershell
.\.venv\Scripts\python.exe -m video_classification.train --config configs/04_cnn3d.toml --epochs 2 --image-size 112
```

Omit `--epochs 2 --image-size 112` for up to 20 epochs at 224 pixels.
Runs go to new timestamped folders under `outputs/04_cnn3d_seed42/`, with
video training metrics, validation metrics, and a reloadable best checkpoint.
Use matching resolutions and frame sampling when comparing all four models.

## Controlled normalization trials

The [training review](TRAINING_REVIEW.md) records the current baseline results,
the measured pilot outcomes, and the reasoning behind these trials.

The optional model setting `normalization = "group"` inserts GroupNorm with
8 groups after each convolution and before ReLU. It normalizes feature groups
within each sample, so it works with the current batches of 1–2 videos.
The convolution widths and fusion methods stay the same. The default is
`"none"`; existing configurations and checkpoints retain their original behavior.
This is a candidate improvement to measure, not a guaranteed accuracy gain.

Four configurations under `configs/tuning/` enable only this architecture
change. They use 112-pixel inputs, seed 42, and the baseline learning rate and
batch sizes, matching the laptop experiments. A short trial is:

```powershell
.\.venv\Scripts\python.exe -m video_classification.train --config configs/tuning/02_late_fusion_groupnorm.toml --epochs 5
```

Compare it with the first five epochs of the baseline at 112 pixels. Omit
`--epochs 5` for a trial of up to 20 epochs. Use `--patience 0` when matching
the full 20-epoch budget of the historical baselines. Corresponding files exist for per-frame,
early fusion, and 3D CNN. Runs are saved under `outputs/tuning/`.
These trials start from scratch; they do not resume your trained checkpoints.
The saved model configuration determines whether normalization is used on reload.

For cluster experiments, copy the selected configuration, preserve its model
options, and measure runtime/memory before increasing batch size or resolution.
Compare one change at a time and record the number of optimizer updates as well
as epochs: per-frame training sees 5,000 frames per epoch, whereas clip training
sees 500 videos. Use the same validation protocol and reserve the test split for
final evaluation.

See the [Group Normalization paper](https://openaccess.thecvf.com/content_ECCV_2018/html/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.html)
for the motivation for normalization with small batches.

## Initialize late fusion from your trained per-frame model

Late fusion shares all encoder and classifier parameters with the per-frame
model. You can reuse learned weights before fine-tuning with a video-level loss.
The `configs/tuning/02_late_fusion_warmstart.toml` configuration points to the
existing per-frame best checkpoint and uses a smaller learning rate of `0.0001`.

```powershell
.\.venv\Scripts\python.exe -m video_classification.train --config configs/tuning/02_late_fusion_warmstart.toml --epochs 2
```

The command starts from the per-frame weights every time, creates a fresh
optimizer, and writes a new run under `outputs/tuning/`. Omit `--epochs 2`
for up to 10 fine-tuning epochs. This is initialization, not resume; it does not
continue from a previous late-fusion fine-tuning run.

Use `--init-checkpoint PATH` or `training.init_checkpoint` to select another
per-frame checkpoint. On the cluster, copy that checkpoint too and set its
project-relative path in the config. The source must have matching class labels,
normalization, and dataset regime. Original-split checkpoints cannot initialize
the corrected-split experiment. Smoke-test checkpoints are rejected.

The checkpoint's source path, SHA-256, and epoch are recorded in
`environment.json` and `best.pt`; the source file is never overwritten.
For the report, describe this as per-frame pretraining followed by video
fine-tuning, with both training budgets recorded. It is a separate experiment
from training late fusion from scratch.
