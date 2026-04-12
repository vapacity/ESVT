#!/bin/bash

# Train MvHeatDET on UAV-EOD dataset
# Usage: bash train_uaveod.sh

cd /Users/zwj/Documents/毕设/OpenEvDET/MvHeatDET

python tools/train.py \
    -c configs/evheat/MvHeatDET_UAVEOD.yml \
    --use-amp \
    --seed 0
