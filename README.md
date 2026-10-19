# Video classification — 2026

Project scaffold for `Project_2_VideoClassification.pdf`. The assignment PDF is
the source of requirements. The 2025 project has not been copied or modified.

## Status

The package layout, experiment configurations, configuration checker, notebook,
and report outline are ready. Data loading, model architectures, training, and
evaluation are **implementation tasks**, with interfaces and TODOs in `src/`.
No datasets, pretrained weights, training results, or Python environment are
included. No training has been run.

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
    data.py                  Course loader integration interface
    models/                  Four single-stream approaches and dual stream
    train.py                 Training implementation starting point
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
initially names `per_frame`; **replace it with the validation-selected model and
matching hyperparameters** from 4.1 before the comparison. Never choose using test
accuracy. The corrected split sizes are not assumed to match the original data.

The original dataset has 10 classes, 10 sampled RGB frames per video, and
500/120/120 train/validation/test videos. Do not infer the numeric class mapping
from the order of names in the brief; read it from the provided CSVs.

## Implementation order

1. Complete `data.py` using the course loader; verify split membership and labels.
2. Implement the four models in `models/`, then the training and evaluation loops.
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
