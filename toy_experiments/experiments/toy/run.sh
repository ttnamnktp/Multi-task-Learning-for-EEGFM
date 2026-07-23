#!/bin/bash
# Tự xác định thư mục gốc toy_experiments dù script được gọi từ đâu
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR/../../"   # experiments/toy -> lên 2 cấp = toy_experiments

cd "$ROOT_DIR"
mkdir -p ./trainlogs

method=famo
seed=42
gamma=0.0
scale=0.1
method_params_lr=0.5

python -m experiments.toy.trainer --n-epochs=50000 --scale=$scale --method=$method --seed=$seed --gamma=$gamma --method-params-lr=$method_params_lr \
    > trainlogs/famo-gamma$gamma-$seed.log 2>&1 &