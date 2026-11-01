#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-configs/ppo_cloud.toml}"
INIT="${INIT_CHECKPOINT:-}"

echo "[cloud] config=$CONFIG"
python - <<'PY'
import taverngym, torch
print("taverngym", taverngym.version(), taverngym.build_info())
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
PY

if [[ -n "$INIT" ]]; then
  python scripts/train_ppo.py --config "$CONFIG" --init-checkpoint "$INIT"
else
  python scripts/train_ppo.py --config "$CONFIG"
fi

echo "[cloud] done. see runs/ under output_dir in the toml"
