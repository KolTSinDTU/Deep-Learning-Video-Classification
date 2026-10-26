#!/bin/bash
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J cnn3d
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=5GB]"
#BSUB -W 12:00
#BSUB -o logs/cnn3d_%J.out
#BSUB -e logs/cnn3d_%J.err

set -e
cd "$LS_SUBCWD"
module load "${VC_PYTHON_MODULE:?Set the Python module in jobs/submit_all.sh}"
source .venv/bin/activate
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python -u -m video_classification.train --config configs/04_cnn3d.toml --device cuda
