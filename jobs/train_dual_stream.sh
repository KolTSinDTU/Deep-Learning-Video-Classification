#!/bin/bash
#BSUB -q c02516
#BSUB -app c02516_1g.10gb
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J dual_stream
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=5GB]"
#BSUB -W 12:00
#BSUB -o logs/dual_stream_%J.out
#BSUB -e logs/dual_stream_%J.err

# Submit with one command (the wrapper creates logs/ and invokes bsub):
#   bash jobs/submit_dual_stream.sh
# Optional short check: bash jobs/submit_dual_stream.sh smoke
# Use the same loaded modules as your working .venv. To explicitly load Python,
# export VC_PYTHON_MODULE with that module's name before submitting.

set -eo pipefail
cd "${LS_SUBCWD:?Submit with bsub from the project directory}"
test -f configs/06_dual_stream.toml

if [[ -n "${VC_PYTHON_MODULE:-}" ]]; then
    module load "$VC_PYTHON_MODULE"
elif [[ -f jobs/modules.sh ]]; then
    source jobs/modules.sh
fi
source .venv/bin/activate
set -u

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

extra=()
case "${VC_RUN_MODE:-train}" in
    smoke) extra+=(--smoke-test) ;;
    pilot) extra+=(--epochs 2) ;;
    train) ;;
    *) echo "VC_RUN_MODE must be smoke, pilot or train." >&2; exit 2 ;;
esac

echo "Job ${LSB_JOBID}: dual_stream (${VC_RUN_MODE:-train})"
hostname
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print("Torch:", torch.__version__, "GPU:", torch.cuda.get_device_name(0))'

exec python -u -m video_classification.train \
    --config configs/06_dual_stream.toml \
    --device cuda \
    "${extra[@]}"
