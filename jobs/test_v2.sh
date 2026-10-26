#!/bin/bash
#BSUB -q c02516
#BSUB -app c02516_1g.10gb
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J v2_test
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=5GB]"
#BSUB -W 01:00
#BSUB -o logs/v2_test_%J.out
#BSUB -e logs/v2_test_%J.err

# Final test evaluation of the corrected-split v2 runs, plus two-stream fusion.
# Submit from the project directory:  bsub < jobs/test_v2.sh
# Writes eval_val.json/eval_test.json next to each best.pt and
# outputs/v2/fusion/seed42/fusion_val_test.json. Optional: VC_SEED (default 42).

set -eo pipefail
cd "${LS_SUBCWD:?Submit with bsub from the project directory}"
if [[ -n "${VC_PYTHON_MODULE:-}" ]]; then
    module load "$VC_PYTHON_MODULE"
elif [[ -f jobs/modules.sh ]]; then
    source jobs/modules.sh
fi
source .venv/bin/activate
set -u
seed="${VC_SEED:-42}"

newest() {  # newest run that completed all epochs; ignores killed and smoke runs
    find "$1" -mindepth 1 -maxdepth 1 -type d ! -name 'smoke_*' \
        -exec test -f '{}/best.pt' ';' -exec grep -q '"stop_reason": "max_epochs"' '{}/summary.json' ';' -print | sort | tail -n 1
}

for experiment in 05_per_frame_noleak 06_late_fusion_noleak 09_early_fusion_noleak \
                  10_cnn3d_noleak 07_flow_stream 08_dual_stream_joint; do
    run=$(newest "outputs/v2/$experiment/seed$seed")
    if [[ -z "$run" ]]; then
        echo "SKIP $experiment: no completed run" >&2
        continue
    fi
    for split in val test; do
        python -u -m video_classification.evaluate --checkpoint "$run/best.pt" --split "$split" --device cuda
    done
done

rgb=$(newest "outputs/v2/06_late_fusion_noleak/seed$seed")
flow=$(newest "outputs/v2/07_flow_stream/seed$seed")
python -u -m video_classification.fuse --rgb "$rgb/best.pt" --flow "$flow/best.pt" \
    --output-dir "outputs/v2/fusion/seed$seed" --test --device cuda
