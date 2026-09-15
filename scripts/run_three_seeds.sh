#!/usr/bin/env bash
set -euo pipefail
CFG=${1:-configs/dvlog.yaml}
NAME=${2:-dvlog}
for SEED in 1 2 3; do
  python train.py --config "$CFG" --seed "$SEED" --output "runs/${NAME}/seed${SEED}"
done
