# Neighbor-Aware Snapshot-Based Temporal Graph Learning (NAGNN)

This repository contains the implementation of NAGNN, a snapshot-based method for temporal graph learning.

## Environment

The project uses the UTG environment described in `environment.yml` (Python 3.11, PyTorch 2.5.1 with CUDA 12.1, PyG 2.6.1, and TGB/TGX).

```bash
conda env create -n utg -f environment.yml
conda activate utg
```

## Run

Run commands from the repository root. Dataset files and negative samples should be placed under `data/`; data and generated artifacts are excluded from Git.

```bash
# Snapshot-based experiments
python dtdg_nagnn.py -d uci -t weekly --model NAGNN --device_id 0

# TGB experiments
python ctdg_nagnn.py -d tgbl-wiki -t hourly --model NAGNN --device_id 0
```
