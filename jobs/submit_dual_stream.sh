#!/bin/bash
# Run on the cluster: bash jobs/submit_dual_stream.sh [train|smoke|pilot]
# Defaults to submitting full training. No GPU allocation is needed to submit.
set -euo pipefail

if (( $# > 1 )); then
    echo "Usage: bash jobs/submit_dual_stream.sh [train|smoke|pilot]" >&2
    exit 2
fi
run_mode="${1:-train}"
submit_options=()
case "$run_mode" in
    train|pilot) ;;
    smoke) submit_options+=(-W 00:30) ;;
    *) echo "Mode must be train, smoke or pilot." >&2; exit 2 ;;
esac

cd "$(dirname "${BASH_SOURCE[0]}")/.."
if ! command -v bsub >/dev/null 2>&1; then
    echo "bsub is unavailable. Run this script in your cluster submission shell." >&2
    exit 1
fi
test -f jobs/train_dual_stream.sh
test -f configs/06_dual_stream.toml
mkdir -p logs

echo "Submitting dual-stream ${run_mode} job; logs: logs/dual_stream_<job-id>.out and .err"
exec bsub -env "all,VC_RUN_MODE=$run_mode" "${submit_options[@]}" < jobs/train_dual_stream.sh
