#!/bin/bash
#BSUB -q c02516
#BSUB -app c02516_1g.10gb
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=5GB]"
#BSUB -W 04:00

# Generic v2 training job; submitted by jobs/submit_v2.sh, which sets:
#   VC_CONFIG     configs/v2/<name>.toml
#   VC_OUTPUT     parent directory for this run's timestamped folder
#   VC_SEED       experiment seed
#   VC_INIT_FROM  optional: a per-frame output directory; its newest
#                 non-smoke best.pt initializes this model
#   VC_RUN_MODE   train or smoke

set -eo pipefail
cd "${LS_SUBCWD:?Submit with bsub from the project directory}"
test -f "${VC_CONFIG:?VC_CONFIG is required}"

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

extra=(--seed "${VC_SEED:-42}" --output-dir "${VC_OUTPUT:?VC_OUTPUT is required}")
if [[ "${VC_RUN_MODE:-train}" == "smoke" ]]; then
    extra+=(--smoke-test)
fi
if [[ -n "${VC_INIT_FROM:-}" ]]; then
    # Run folders are UTC timestamps, so lexical order is chronological.
    # Runs that were killed or are still running have no "max_epochs" stop reason.
    source_run=$(find "$VC_INIT_FROM" -mindepth 1 -maxdepth 1 -type d ! -name 'smoke_*' \
                 -exec test -f '{}/best.pt' ';' -exec grep -q '"stop_reason": "max_epochs"' '{}/summary.json' ';' -print | sort | tail -n 1)
    if [[ -z "$source_run" ]]; then
        echo "No completed per-frame run (best.pt, all epochs) in $VC_INIT_FROM" >&2
        exit 1
    fi
    extra+=(--init-checkpoint "$source_run/best.pt")
fi

echo "Job ${LSB_JOBID}: ${VC_CONFIG} seed ${VC_SEED:-42} (${VC_RUN_MODE:-train})"
hostname
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print("Torch:", torch.__version__, "GPU:", torch.cuda.get_device_name(0))'

exec python -u -m video_classification.train --config "$VC_CONFIG" --device cuda "${extra[@]}"
