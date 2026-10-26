**Train the 2026 video-classification project on DTU HPC**

Prepared 19 October 2026 from the HPC slides, the assignment, the current project code, and DTU's public documentation. This guide assumes you mean the current `2026/Video Classification` project. Cluster commands are instructions for you to run; I have not logged into your account or submitted jobs.

The workflow is: upload code, create a Linux environment, connect the dataset, check the GPU, submit small batch checks, then submit full training jobs. Your laptop can disconnect after a batch job has been accepted.

**Native-resolution preparation (required before the training commands below).**

Configurations now read full-resolution PNGs from `data/native_frames/original`
using `frame_source = "native_video"` and `image_size = "native"`. The supplied
JPEGs are already 224 x 224; the original train/validation AVI files are 320 x 240.
After setting up the Python environment, make FFmpeg and ffprobe available on
PATH and run this once on an allocated compute node, from the project directory:

```bash
python -m video_classification.prepare_frames --root /dtu/datasets1/02516/ufc10 --output-root data/native_frames/original --splits train val --workers 4
```

The cache is stored in your writable project directory, not the shared dataset.
The command preserves source files, samples 10 evenly spaced decoded frames per
video, records the indices, and reuses completed matching caches on subsequent
runs. These are newly sampled clips: the original JPEG sampling indices are
unknown. No test videos are read. Keep the same cache for all four comparisons.
For corrected splits, use the verified corrected dataset root and
`--output-root data/native_frames/no_leakage` before experiment 5. Experiment 6
uses supplied RGB frames to match the supplied flow. Native dual-stream input
requires flow for the exact cached frame pairs and matching source-index
metadata; see `data/README.md`. Match the RGB-only comparison's preprocessing.

Run the smoke checks again at native resolution before full training: 320 x 240
has about 1.53 times as many pixels as 224 x 224, increasing activation memory
and computation. Upload the new source and configurations together.

**1. Know which experiments you can run now.**

| Array index | Configuration | Current status |
| --- | --- | --- |
| 1 | `configs/01_per_frame.toml` | Implemented; up to 50 epochs, native 320 x 240 frames, batch of 8 frames |
| 2 | `configs/02_late_fusion.toml` | Implemented; up to 50 epochs, native 320 x 240 frames, batch of 2 videos |
| 3 | `configs/03_early_fusion.toml` | Implemented; up to 50 epochs, native 320 x 240 frames, batch of 2 videos |
| 4 | `configs/04_cnn3d.toml` | Implemented; up to 50 epochs, native 320 x 240 frames, batch of 1 video |
| 5 | `configs/05_no_leakage.toml` | Configure after selecting a model from 1–4; check corrected data compatibility |
| 6 | `configs/06_dual_stream.toml` | Implemented; reads supplied `flows_png` directly; batch of 1 video |

The current code includes the 3D CNN. Standalone checkpoint/test evaluation in `evaluate.py` is unfinished. Training already computes validation metrics and saves checkpoints. Epoch counts are upper limits: validation-loss early stopping defaults to patience 5 and a minimum decrease of 0.0001. Configure `early_stopping_patience` and `early_stopping_min_delta` under `[training]`, or pass `--patience` and `--min-delta`. Set patience to 0 for a fixed-epoch comparison.

**2. Connect from Windows.**

Connect to the DTU network on campus, or connect to DTU VPN first. Open PowerShell and replace `sXXXXXX` with your DTU username:

```powershell
ssh sXXXXXX@login1.hpc.dtu.dk
```

Use your DTU credentials. On the first connection, compare the displayed host fingerprint with [DTU's fingerprints](https://www.hpc.dtu.dk/fp.txt). Password characters do not appear while typing. Access without VPN from outside DTU requires the SSH-key setup described in [DTU's access instructions](https://www.hpc.dtu.dk/?page_id=4317).

The login node is an entry point. Run training on allocated compute resources. The commands below explicitly say which terminal to use. [DTU login documentation](https://www.hpc.dtu.dk/?page_id=2501).

**3. Copy the code to your cluster home directory.**

In a SECOND local PowerShell tab, run:

```powershell
Set-Location 'C:\Users\kolbe\Desktop\Skóli\DTU\Intro to deep learning in computer vision\2026\Video Classification'
$clusterTarget = 'sXXXXXX@login1.hpc.dtu.dk'
ssh $clusterTarget 'mkdir -p projects/video-classification-2026'
scp -r src configs pyproject.toml README.md "${clusterTarget}:projects/video-classification-2026/"
```

This copies your current source files, including local changes. The cluster project directory will be `~/projects/video-classification-2026`. Build a new Linux virtual environment there; your Windows `.venv` cannot be reused. The shared course dataset usually makes a data upload unnecessary.

Keep the local PowerShell tab open for downloading results later. After making future code changes, repeat the `scp` command when no jobs are running or waiting to use those files. Queued jobs read the files when they start.

**4. Open the course's interactive GPU shell for setup and a short check.**

In the SSH terminal:

```bash
02516sh
cd ~/projects/video-classification-2026
mkdir -p jobs logs outputs
hostname
nvidia-smi
getquota_zhome.sh
```

Wait for the interactive shell to open before continuing. The slides allow one `02516sh` session at a time. Use `bstat` to find an existing session if necessary, and `exit` to release it when finished. If an abandoned session must be cancelled, use `bkill JOBID` for that specific session.

Use this session briefly for setup/debugging. Full training goes through batch jobs. The interactive GPUs and the batch GPU slices can have different memory capacity, so an interactive success alone does not prove a model fits the 10 GB batch allocation. This distinction comes from slides 23–24 and 43.

**5. Create a compatible Python environment.**

Still inside `02516sh`, list the available Python modules:

```bash
module avail python3
```

Choose an available Python 3.11, 3.12 or 3.13 module; 3.12 is a suitable choice. Paste its EXACT name when prompted below, for example `python3/3.12.11` only if it appears in the list:

```bash
read -r -p 'Exact Python module name: ' VC_PY_MODULE
module load "$VC_PY_MODULE"
python3 --version
python3 -c 'import sys; assert (3,11) <= sys.version_info[:2] < (3,14), sys.version'
printf 'module load %s\n' "$VC_PY_MODULE" > jobs/modules.sh
ldd --version | head -n 1
```

Check that these commands succeed before continuing. The module name is saved so batch jobs load the same Python. DTU requires loading the original module whenever reusing a virtual environment. [DTU Python environments](https://www.hpc.dtu.dk/?page_id=3678).

The project pins `torch==2.8.0` and `torchvision==0.23.0`. Their Linux x86-64 wheels require glibc 2.28 or newer. If the allocated node reports an older glibc, obtain a compatible course environment/container or a validated alternative dependency pair before continuing; loading a newer Python module does not by itself upgrade glibc. The public DTU login page still mentions Scientific Linux 7.9, so the actual compute-node check matters. [PyTorch 2.8 wheel requirements](https://pypi.org/project/torch/2.8.0/).

On a compatible node, install the CUDA 12.6 build of the pinned pair:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install --no-cache-dir -e .
python -m pip check
python -m pip freeze > jobs/environment-freeze.txt
```

For limited storage, use the updated `pyproject.toml`: the default install now
contains only the five direct training dependencies and their required supporting
packages. Analysis and notebook packages are optional. Re-upload this file if you
copied the project before the dependency change. All models share the same `.venv`.
Avoid installing `.[notebooks]` for batch training.

Inspect storage with `du -sh .venv` and `python -m pip cache info`. Clear old pip
downloads with `python -m pip cache purge`; this does not uninstall packages.
The `--no-cache-dir` flags above prevent retaining new download-cache copies.
Previously installed optional packages are not removed by reinstalling the
project; after updating its metadata with `pip install -e .`, they can be
uninstalled if no longer needed. Keep PyTorch's required NVIDIA runtime packages.

This CUDA build is published for the project's pinned versions; driver compatibility still needs the GPU check in step 7. If installation fails, resolve it before submitting training. Do not install the CPU-only build for these GPU jobs. [Official PyTorch version/install matrix](https://pytorch.org/get-started/previous-versions/#v280).

**6. Connect the course data to the paths your configurations expect.**

Inspect the shared course directory:

```bash
ls /dtu/datasets1/02516
find /dtu/datasets1/02516 -maxdepth 4 -type f -path '*/metadata/train.csv'
```

There is a naming inconsistency in the supplied material: the assignment calls the corrected dataset `ucf101noleakage`, while `src/datasets.py` uses `ucf101_noleakage`. The loader identifies the original root as `/dtu/datasets1/02516/ufc10`. Use the directory you actually find, containing both `metadata/` and `frames/`. The HPC slides' generic `/02514/` example is not this assignment's dataset location.

For the original dataset, if the following checks succeed:

```bash
test -f /dtu/datasets1/02516/ufc10/metadata/train.csv && echo 'Training CSV found'
test -d /dtu/datasets1/02516/ufc10/frames/train && echo 'Training frames found'
```

Create a link at the path already used by configurations 1–4:

```bash
mkdir -p data/ucf101
ln -s /dtu/datasets1/02516/ufc10 data/ucf101/ufc10
```

Run the link command once, with an absent destination. If the shared root is different, substitute the confirmed path. If you already have a local directory/link there, inspect it instead of overwriting it.

If the original data is unavailable on the cluster, upload only your local extracted frames and metadata. First create the destination from the local PowerShell tab:

```powershell
ssh $clusterTarget 'mkdir -p projects/video-classification-2026/data/ucf101/ufc10'
scp -r data/ucf101/ufc10/frames data/ucf101/ufc10/metadata "${clusterTarget}:projects/video-classification-2026/data/ucf101/ufc10/"
```

This fallback assumes `data/ucf101/ufc10` is a real writable directory, not a link into the read-only shared dataset. Raw `videos/` are unnecessary for the four current models.

Verify the configuration in the interactive shell:

```bash
python -m video_classification.cli --config configs/01_per_frame.toml --require-data
head -n 3 data/ucf101/ufc10/metadata/train.csv
```

The configuration checker only checks the root directory. The training adapter expects `metadata/train.csv`, `metadata/val.csv`, CSV columns `video_name,video_path,action,label`, and frames named `frame_1.jpg` through `frame_10.jpg` under `frames/<split>/<relative-video-path-without-extension>/`. The smoke test will check the actual data layout.

**7. Confirm CUDA works, then run one tiny interactive check.**

Inside the same allocated GPU shell and active environment:

```bash
python - <<'PY'
import torch, torchvision
print('Torch:', torch.__version__, 'Torchvision:', torchvision.__version__)
print('Built for CUDA:', torch.version.cuda)
assert torch.cuda.is_available(), 'CUDA unavailable: check the allocated GPU and installed build'
print('GPU:', torch.cuda.get_device_name(0))
x = torch.randn(32, 32, device='cuda')
print('CUDA calculation:', (x @ x).sum().item())
PY

python -m video_classification.train --config configs/01_per_frame.toml --smoke-test --image-size 112 --device cuda
```

Require both commands to succeed. The small training check runs at most two training and two validation batches and saves a separately marked smoke run. Its scores are not experimental results.

Use `--device cuda` in cluster jobs so a broken GPU setup produces an error instead of silently training on the CPU. Let the scheduler manage GPU visibility; do not copy the slide's `CUDA_VISIBLE_DEVICES=1` example into a scheduled job.

**8. Create a reusable batch script.**

Still in the project directory, paste this entire block into the HPC shell. It creates a Bash script with Linux line endings:

```bash
cat > jobs/train_models.lsf <<'EOF'
#!/bin/bash
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J "video_train[1-4]%1"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=5GB]"
#BSUB -W 12:00
#BSUB -o logs/train_%J_%I.out
#BSUB -e logs/train_%J_%I.err

set -eo pipefail
cd "$HOME/projects/video-classification-2026"
source jobs/modules.sh
source .venv/bin/activate
set -u

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

case "${LSB_JOBINDEX:-0}" in
  1) config=configs/01_per_frame.toml ;;
  2) config=configs/02_late_fusion.toml ;;
  3) config=configs/03_early_fusion.toml ;;
  4) config=configs/04_cnn3d.toml ;;
  5) config=configs/05_no_leakage.toml ;;
  6) config=configs/06_dual_stream.toml ;;
  *) echo 'Submit as an array using an index from 1 through 6.' >&2; exit 2 ;;
esac

echo "Job ${LSB_JOBID}, index ${LSB_JOBINDEX}, configuration ${config}"
hostname
module list
nvidia-smi
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'

extra=()
case "${VC_RUN_MODE:-train}" in
  smoke) extra+=(--smoke-test) ;;
  pilot) extra+=(--epochs 2) ;;
  train) ;;
  *) echo 'VC_RUN_MODE must be smoke, pilot or train.' >&2; exit 2 ;;
esac

python -u -m video_classification.train --config "$config" --device cuda "${extra[@]}"
EOF

bash -n jobs/train_models.lsf
```

The queue, GPU application profile in the next step, and 12-hour maximum come from slides 36–38. The template corrects the slide's curly quotation marks and puts `#BSUB -W 12:00` on its own line; on the slide, it appears after another comment and would not be applied if copied literally.

The CPU memory request here is a suggested starting point: DTU documents `rusage[mem=...]` per CPU slot, so 4 slots × 5 GB reserves 20 GB of host RAM. GPU memory is separate: the course application profile requests the 10 GB GPU slice. Adjust host RAM based on measured use. [DTU batch resource semantics](https://www.hpc.dtu.dk/?page_id=1416).

The array runs one model per element. `%1` limits this array to one running element at a time; all four can be queued together. Each element has its own time limit and logs. Indices do not guarantee execution order, which is fine for these independent baselines. Avoid overlapping arrays or group members' submissions if you are sharing the course allocation. [IBM job arrays](https://www.ibm.com/docs/en/spectrum-lsf/10.1.0?topic=administration-job-arrays).

**9. Release the interactive GPU and check all four models on the actual batch allocation.**

Exit `02516sh` once. From the login shell, enter a general application shell for preparing/submitting jobs:

```bash
exit
linuxsh
cd ~/projects/video-classification-2026
mkdir -p logs
bsub -app c02516_1g.10gb -J "video_smoke[1-4]%1" -W 00:30 -env "all,VC_RUN_MODE=smoke" < jobs/train_models.lsf
```

The returned number is the array job ID; save it. This checks all four models at the configured native 320 x 240 resolution and batch sizes on the 10 GB allocation. Wait for ALL FOUR elements to finish successfully before full training. An array can contain both successful and failed elements.

The `-env` argument selects smoke mode explicitly for the batch environment. [IBM environment passing](https://www.ibm.com/docs/en/spectrum-lsf/10.1.0?topic=o-env).

If the course queue or application profile is rejected, check your course access and the current instructions with the teaching team; these identifiers come from the supplied 2026 slides, not a live query of your account.

**10. Measure runtime, then submit the four full experiments.**

A two-epoch pilot measures complete train/validation passes. The smoke test only processes a few batches and cannot estimate full training time accurately. After all smoke jobs finish, submit:

```bash
bsub -app c02516_1g.10gb -J "video_pilot[1-4]%1" -env "all,VC_RUN_MODE=pilot" < jobs/train_models.lsf
```

Read each pilot's per-epoch `seconds` in its `history.csv`. Estimate `20 × typical epoch seconds`, allowing extra time for startup and saving. If that fits within 12 hours per model, submit the final configured runs after the pilots complete:

```bash
bsub -app c02516_1g.10gb -J "video_train[1-4]%1" -env "all,VC_RUN_MODE=train" < jobs/train_models.lsf
```

Each model starts from fresh weights and uses its TOML settings. Pilot runs also start fresh and are ordinary timestamped runs; identify them by `config.json` showing 2 epochs, and keep them out of the final 20-epoch comparison.

Once accepted, these batch jobs continue when you close PowerShell or disconnect your laptop. Do not launch the script with `bash jobs/train_models.lsf` for training: that bypasses scheduling.

If memory is exhausted, lower the affected configuration's `training.batch_size` first. The 3D model already starts at 1; if needed, lower resolution and use a consistent comparison protocol across models. Record any changes and repeat the affected smoke check. Keep validation batch size at 1 initially. Leave `num_workers=0` for the first runs; tune data loading only after the pipeline works.

If a full run would exceed 12 hours, plan a shorter experiment or implement/test checkpoint resume before submitting it. The current trainer saves a best checkpoint but has NO resume command. Rerunning starts a new experiment; a timeout does not automatically continue the old one.

**11. Monitor jobs and inspect results.**

From the HPC application shell:

```bash
bstat
bjobs -a
```

Replace `123456` below with your array job ID. Index 1 is the per-frame model:

```bash
bjobs -l '123456[1]'
bpeek '123456[1]'
bhist -l '123456[1]'
```

`PEND` means queued, `RUN` means running, `DONE` means the job completed successfully, and `EXIT` means it ended unsuccessfully. Check `bpeek` during execution; output files may not be fully written until completion. [DTU monitoring commands](https://www.hpc.dtu.dk/?page_id=1519).

Logs use names such as `logs/train_123456_1.out` and `logs/train_123456_1.err`. Cancel only a specific unwanted element with `bkill '123456[1]'`; `bkill 123456` cancels the entire array.

The trainer prints a unique output directory for each invocation, for example:

```text
outputs/01_per_frame_seed42/<timestamp>/
    best.pt
    best_loss.pt
    history.csv
    summary.json
    config.json
    class_mapping.json
    environment.json
```

For each final model, check that the saved configuration is intended and that `summary.json` reports either `max_epochs` or `early_stopping`. A completed early-stopped run can have fewer epochs than requested; a summary still marked `running` is not confirmation of normal completion. `best_loss.pt` is the lowest-validation-loss checkpoint for loss-based selection, while `best.pt` retains highest validation video accuracy, breaking ties by loss. Both can come from earlier epochs. Exclude `smoke_*` directories and the two-epoch pilots from the final comparison. No test data is read during training.

**12. Retrain the selected model on the corrected dataset.**

Choose a well-performing model from configurations 1–4 using validation results. In `configs/05_no_leakage.toml`, replace the default `model.name = "per_frame"` with the chosen name and copy its relevant model/training settings, including batch size, learning rate, frame count and image size. Preserve experiment 5's name, output directory and `regime = "no_leakage"`.

After confirming the real corrected-data path from step 6, link it to the expected location. This example applies ONLY if the underscored directory exists and the link destination is absent:

```bash
ln -s /dtu/datasets1/02516/ucf101_noleakage data/ucf101noleakage
```

Inspect its metadata and frames before running. Setting `regime = "no_leakage"` is only a configuration label; it does not repair splits or convert a different data layout. If the corrected data uses different filenames/columns, adapt the loader first while preserving the provided splits and labels.

Then submit only element 5, first as a smoke check, and submit the full run only after that check succeeds:

```bash
bsub -app c02516_1g.10gb -J "corrected_smoke[5]%1" -W 00:30 -env "all,VC_RUN_MODE=smoke" < jobs/train_models.lsf
```

```bash
bsub -app c02516_1g.10gb -J "corrected_train[5]%1" -env "all,VC_RUN_MODE=train" < jobs/train_models.lsf
```

This retrains the chosen architecture from scratch on the corrected training split, as requested by the assignment.

**13. Prepare dual-stream data and complete final evaluation.**

Configuration 6 now includes the network, paired loader, and training support.
The confirmed cluster root is `/dtu/datasets1/02516/ucf101_noleakage`. Set
`dataset.root` to that directory or use the local symlink from step 12. The loader
reads `flows_png/<split>/<class>/<video>/flow_1_2.png` through `flow_9_10.png`
directly; no pairing manifests are needed for this layout. Check the PNG channel
format and dimensions in a smoke test. Use `rgb_png` for RGB visualizations or
`uv_npy` with an explicit `flow_scale` for vector arrays. Keep supplied flow paired
with supplied RGB frames. Run configuration 6 with `--smoke-test` before adding
index 6 to the batch template. The local pipeline checks use synthetic data;
real corrected-data validation and standalone final test evaluation remain.
Custom vector data or native-cache flow requires `flow_manifest_root`; see
`data/README.md` for that optional schema.

The dedicated `jobs/train_dual_stream.sh` job uses the same course queue and
GPU profile as the other jobs. Copy the updated source, configuration, and both
`jobs/train_dual_stream.sh` and `jobs/submit_dual_stream.sh` to the cluster.
Submit full training with one command from the project directory:

```bash
bash jobs/submit_dual_stream.sh
```

The submission script creates `logs/`, invokes `bsub`, and prints the job ID.
It can also be invoked by absolute path from another directory. To submit a
30-minute smoke check first, use:

```bash
bash jobs/submit_dual_stream.sh smoke
```

The script activates `.venv` and uses the loaded module environment. If set,
`VC_PYTHON_MODULE` loads that Python module; otherwise it sources
`jobs/modules.sh` when present. Use the same modules as your working environment.
Logs are `logs/dual_stream_<job-id>.out` and `.err`; the trainer prints its new
run directory in the output log. `bash jobs/submit_dual_stream.sh pilot` submits
two full epochs. Run the submission script with `bash`; do not directly run
`jobs/train_dual_stream.sh`, which is the scheduled job payload.

For final reporting, implement checkpoint loading and video-level evaluation in `evaluate.py`. Restore the saved configuration and class mapping, evaluate each selected best checkpoint in evaluation/inference mode, and report test metrics for the appropriate dataset regime. Use validation for model/hyperparameter selection and reserve test data for final reporting. The current repository has no working final-evaluation CLI to invoke.

**14. Download the checkpoints, metrics and logs.**

After the jobs finish, use your LOCAL PowerShell tab, from the local project directory:

```powershell
$clusterTarget = 'sXXXXXX@login1.hpc.dtu.dk'
$downloadFolder = 'hpc-results-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
New-Item -ItemType Directory -Path $downloadFolder
scp -r "${clusterTarget}:projects/video-classification-2026/outputs" $downloadFolder
scp -r "${clusterTarget}:projects/video-classification-2026/logs" $downloadFolder
scp -r "${clusterTarget}:projects/video-classification-2026/jobs" $downloadFolder
scp -r "${clusterTarget}:projects/video-classification-2026/configs" $downloadFolder
```

Keep these results together with the exact source code used. Avoid committing checkpoints or datasets to Git. On the cluster, `getquota_zhome.sh` checks home-directory usage; download results and verify your copies before deciding which old runs to remove.

**If a step fails**

| Symptom | What to check next |
| --- | --- |
| SSH reports `Permission denied (publickey)` from home | Connect to DTU VPN or complete DTU's off-campus SSH-key procedure |
| `02516sh` will not open another session | Look for your existing interactive job with `bstat` |
| Python module missing | Select an actual 3.11–3.13 module from `module avail python3`; update `jobs/modules.sh` |
| `GLIBC_2.28 not found` or no compatible PyTorch wheel | Check the compute-node OS; obtain a compatible environment or a tested dependency change |
| CUDA unavailable or driver error | Confirm an allocated GPU, pinned CUDA wheels, and the module used to create the environment |
| Missing CSV/frame | Check the dataset root, underscore spelling, CSV-relative paths and frame naming |
| CUDA out of memory | Reduce batch size; check video validation too; request larger course profiles only if actually needed |
| `TERM_RUNLIMIT` | The job exceeded wall time; estimate from pilot epochs and address the lack of resume support |
| `TERM_MEMLIMIT` | Host RAM limit, distinct from GPU memory; inspect peak use and adjust the per-slot RAM request |
| Missing flow file | Check `dataset.root` and `flows_png/<split>/<class>/<video>/flow_i_j.png`; omit `flow_manifest_root` for the supplied layout |
| `NotImplementedError` for evaluation | Complete standalone final test evaluation before submitting that work |

**Source material**

- [HPC lecture slides](<C:/Users/kolbe/Desktop/Skóli/DTU/Intro to deep learning in computer vision/IDLCV_L5_HPC.pdf>): slides 18–19 (access), 23–24 (interactive sessions), 27–32 (storage and environments), 35–38 (batch jobs), 43 (resources), 49 (batch training for Project 2).
- [Video-classification assignment](<C:/Users/kolbe/Desktop/Skóli/DTU/Intro to deep learning in computer vision/2026/Video Classification/Project_2_VideoClassification.pdf>): pages 2–3 (the four models, corrected splits, dual stream and final evaluation).
- Current project source and configurations, particularly [training](<C:/Users/kolbe/Desktop/Skóli/DTU/Intro to deep learning in computer vision/2026/Video Classification/src/video_classification/train.py>), [data loading](<C:/Users/kolbe/Desktop/Skóli/DTU/Intro to deep learning in computer vision/2026/Video Classification/src/video_classification/data.py>), [model construction](<C:/Users/kolbe/Desktop/Skóli/DTU/Intro to deep learning in computer vision/2026/Video Classification/src/video_classification/models/__init__.py>), and [dependencies](<C:/Users/kolbe/Desktop/Skóli/DTU/Intro to deep learning in computer vision/2026/Video Classification/pyproject.toml>).

Course-specific queue names, actual module availability, node OS/driver compatibility and dataset paths still need confirmation in your own cluster session. The batch template and memory starting point are adaptations for this project, not a claim of a completed cluster run.
