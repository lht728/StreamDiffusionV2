#!/bin/bash
set -e
cd /data/StreamDiffusionV2
source venv/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER=1
echo "[start] $(date)"
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B
echo "[done] Wan2.1-T2V-1.3B $(date)"
hf download jerryfeng/StreamDiffusionV2 --local-dir ./ckpts --include "wan_causal_dmd_v2v/*"
echo "[done] StreamDiffusionV2 1.3B $(date)"
curl -L https://github.com/madebyollin/taehv/raw/main/taew2_1.pth -o ckpts/taew2_1.pth
echo "[done] TAEHV $(date)"
echo "[ALL DONE] $(date)"
