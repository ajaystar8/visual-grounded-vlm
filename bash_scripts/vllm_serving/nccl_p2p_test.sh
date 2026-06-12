#!/bin/bash
# Diagnose GPU-to-GPU NCCL P2P health on the current node.
# Runs the 2-GPU all-reduce twice: with P2P (may hang -> timeout) and with
# NCCL_P2P_DISABLE=1 (should pass). A hang in the first + pass in the second
# indicates a broken P2P/IOMMU config on this node -> report to RC.

source /home/${USER}/.bashrc
source activate perception_models

echo "=== node: $(hostname) ==="
nvidia-smi topo -m

echo "=== P2P ENABLED (may hang) ==="
NCCL_DEBUG=WARN timeout 40 python scripts/nccl_p2p_test.py
echo "exit=$?"

echo "=== P2P DISABLED ==="
NCCL_P2P_DISABLE=1 NCCL_DEBUG=WARN timeout 40 python scripts/nccl_p2p_test.py
echo "exit=$?"
