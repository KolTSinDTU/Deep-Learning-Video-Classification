#!/bin/bash
# Run on the cluster with: bash jobs/submit_all.sh
set -eo pipefail

# Edit this ONE line to match the Python module used to create .venv.
export VC_PYTHON_MODULE="YOUR_PYTHON_MODULE"
if [[ "$VC_PYTHON_MODULE" == "YOUR_PYTHON_MODULE" ]]; then
    echo "Set VC_PYTHON_MODULE in this script before submitting." >&2
    exit 2
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs

# Queue all four now. Each starts after the previous job ends, even if it fails.
previous_job=""
for model in per_frame late_fusion early_fusion cnn3d; do
    dependency=()
    if [[ -n "$previous_job" ]]; then
        dependency=(-w "ended($previous_job)")
    fi

    echo "Submitting $model"
    submission=$(bsub -app c02516_1g.10gb \
        -env "all,VC_PYTHON_MODULE=$VC_PYTHON_MODULE" \
        "${dependency[@]}" < "jobs/train_${model}.sh")
    echo "$submission"

    previous_job=$(printf '%s\n' "$submission" | awk -F '[<>]' '/Job </ {print $2; exit}')
    if [[ ! "$previous_job" =~ ^[0-9]+$ ]]; then
        echo "Could not read the submitted job ID. Check bstat before resubmitting." >&2
        exit 1
    fi
done

echo "All four jobs submitted. Check progress with bstat."
