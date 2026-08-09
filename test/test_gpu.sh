#!/bin/bash
#SBATCH --job-name=test-gpu
#SBATCH --account=aip-adurand
#SBATCH --time=0-00:10
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --output=test-gpu-%j.out

module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.12 arrow/18.1.0 \
            mpi4py/4.0.3 nccl/2.26.2 opencv cmake httpproxy

source "$SCRATCH/venv/bin/activate"

export PYTHONNOUSERSITE=1
export HF_HOME=/project/6102313/shared/hf_cache
export HF_HUB_CACHE=/project/6102313/shared/hf_cache/hub
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "================ GPU ================"
hostname
nvidia-smi
echo "====================================="

python - <<'PY'
import torch
import transformers
import vllm

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("vllm:", vllm.__version__)
print("CUDA:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:", torch.cuda.get_device_name(i))
PY
