#!/bin/bash
#BSUB -q c02516
#BSUB -app c02516_1g.10gb
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=5GB]"
#BSUB -W 00:30

# Score fusion of the newest RGB and flow runs; submitted by jobs/submit_v2.sh.
#   VC_RGB_FROM, VC_FLOW_FROM  output directories of the two streams
#   VC_OUTPUT                  where fusion_val.json is written
#   VC_TEST=1                  also score the test split (final reporting only)

set -eo pipefail
cd "${LS_SUBCWD:?Submit with bsub from the project directory}"
if [[ -n "${VC_PYTHON_MODULE:-}" ]]; then
    module load "$VC_PYTHON_MODULE"
elif [[ -f jobs/modules.sh ]]; then
    source jobs/modules.sh
fi
source .venv/bin/activate
set -u

newest() {
    find "$1" -mindepth 1 -maxdepth 1 -type d ! -name 'smoke_*' \
        -exec test -f '{}/best.pt' ';' -exec grep -q '"stop_reason": "max_epochs"' '{}/summary.json' ';' -print | sort | tail -n 1
}
rgb=$(newest "${VC_RGB_FROM:?}")
flow=$(newest "${VC_FLOW_FROM:?}")
if [[ -z "$rgb" || -z "$flow" ]]; then
    echo "Missing a completed RGB or flow run" >&2
    exit 1
fi
test_flag=()
if [[ "${VC_TEST:-0}" == "1" ]]; then
    test_flag=(--test)
fi
exec python -u -m video_classification.fuse --rgb "$rgb/best.pt" --flow "$flow/best.pt" \
    --output-dir "${VC_OUTPUT:?}" --device cuda "${test_flag[@]}"
