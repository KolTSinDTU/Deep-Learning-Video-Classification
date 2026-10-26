#!/bin/bash
# Submit the v2 experiments on the cluster:
#   bash jobs/submit_v2.sh [all|original|no_leakage] [train|smoke]
# Optional: SEEDS="42 43 44" bash jobs/submit_v2.sh
# Optional: SCRATCH=1 also trains late/early fusion and the 3D CNN from random
#           initialization (about 40% more compute).
#
# Jobs run one after another. Warm-started models also wait for their
# per-frame source to succeed. Smoke mode runs every config for 2 batches,
# without warm starts or fusion (smoke checkpoints cannot initialize models).
set -euo pipefail

stage="${1:-all}"
mode="${2:-train}"
case "$stage" in all|original|no_leakage) ;; *) echo "Stage: all, original or no_leakage" >&2; exit 2 ;; esac
case "$mode" in train|smoke) ;; *) echo "Mode: train or smoke" >&2; exit 2 ;; esac
seeds="${SEEDS:-42}"
scratch="${SCRATCH:-0}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
command -v bsub >/dev/null || { echo "Run this in your cluster submission shell." >&2; exit 1; }
mkdir -p logs

previous=""
submit() {  # submit <job name> <script> <extra dependency or ""> <env assignments...>
    local name="$1" script="$2" requires="$3"
    shift 3
    local env="all,VC_RUN_MODE=$mode"
    for assignment in "$@"; do env+=",$assignment"; done
    local condition=""
    [[ -n "$previous" ]] && condition="ended($previous)"
    [[ -n "$requires" ]] && condition="${condition:+$condition && }done($requires)"
    local options=(-J "$name" -o "logs/${name}_%J.out" -e "logs/${name}_%J.err" -env "$env")
    [[ -n "$condition" ]] && options+=(-w "$condition")
    local output
    output=$(bsub "${options[@]}" < "$script")
    echo "$output"
    previous=$(printf '%s\n' "$output" | awk -F '[<>]' '/Job </ {print $2; exit}')
    [[ "$previous" =~ ^[0-9]+$ ]] || { echo "Could not read job ID for $name" >&2; exit 1; }
}

train_job() {  # train_job <config name> <seed> <output subdir> <init source dir or ""> <source job or "">
    local config="$1" seed="$2" output="outputs/v2/$1/$3" init="$4" requires="$5"
    local env=("VC_CONFIG=configs/v2/$config.toml" "VC_SEED=$seed" "VC_OUTPUT=$output")
    if [[ "$mode" == "train" && -n "$init" ]]; then
        env+=("VC_INIT_FROM=$init")
    else
        requires=""
    fi
    submit "v2_${config}_$3" jobs/train_v2.sh "$requires" "${env[@]}"
}

for seed in $seeds; do
    if [[ "$stage" != "no_leakage" ]]; then
        # Task 4.1: per-frame, then each video model warm-started from it ...
        train_job 01_per_frame "$seed" "seed$seed" "" ""
        source_job="$previous"
        source_dir="outputs/v2/01_per_frame/seed$seed"
        for config in 02_late_fusion 03_early_fusion 04_cnn3d; do
            train_job "$config" "$seed" "seed$seed" "$source_dir" "$source_job"
        done
        # ... and optionally from scratch, to separate architecture from initialization.
        if [[ "$mode" == "train" && "$scratch" == "1" ]]; then
            for config in 02_late_fusion 03_early_fusion 04_cnn3d; do
                train_job "$config" "$seed" "scratch_seed$seed" "" ""
            done
        fi
    fi
    if [[ "$stage" != "original" ]]; then
        # Task 4.2.1: the same four models and warm starts on the corrected splits ...
        train_job 05_per_frame_noleak "$seed" "seed$seed" "" ""
        source_job="$previous"
        source_dir="outputs/v2/05_per_frame_noleak/seed$seed"
        for config in 06_late_fusion_noleak 09_early_fusion_noleak 10_cnn3d_noleak; do
            train_job "$config" "$seed" "seed$seed" "$source_dir" "$source_job"
        done
        if [[ "$mode" == "train" && "$scratch" == "1" ]]; then
            for config in 06_late_fusion_noleak 09_early_fusion_noleak 10_cnn3d_noleak; do
                train_job "$config" "$seed" "scratch_seed$seed" "" ""
            done
        fi
        # ... and Task 4.2.2: flow stream, joint dual stream, score fusion.
        train_job 07_flow_stream "$seed" "seed$seed" "" ""
        train_job 08_dual_stream_joint "$seed" "seed$seed" "$source_dir" "$source_job"
        if [[ "$mode" == "train" ]]; then
            submit "v2_fusion_s${seed}" jobs/fuse_v2.sh "" \
                "VC_RGB_FROM=outputs/v2/06_late_fusion_noleak/seed$seed" \
                "VC_FLOW_FROM=outputs/v2/07_flow_stream/seed$seed" \
                "VC_OUTPUT=outputs/v2/fusion/seed$seed"
        fi
    fi
done
echo "Submitted. Check progress with bstat; logs are in logs/."
