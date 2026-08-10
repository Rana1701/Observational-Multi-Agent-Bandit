#!/bin/bash
#SBATCH --job-name=model-sweep
#SBATCH --account=aip-adurand
#SBATCH --time=0-22:59
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --output=%x-%j.out

set -euo pipefail

# ── Modules ───────────────────────────────────────────────────────────────────
module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.12 arrow/18.1.0 \
            mpi4py/4.0.3 nccl/2.26.2 opencv cmake httpproxy

# ── Env ───────────────────────────────────────────────────────────────────────
source "$SCRATCH/venv/bin/activate"

export PYTHONNOUSERSITE=1
export HF_HOME=/project/6102313/shared/hf_cache
export HF_HUB_CACHE=/project/6102313/shared/hf_cache/hub
export TRANSFORMERS_CACHE=/project/6102313/shared/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export COMET_MODE=DISABLED
export HF_DATASETS_CACHE=$SCRATCH/hf_datasets_cache
export TORCH_HOME=$SCRATCH/torch_cache
export TORCHINDUCTOR_CACHE_DIR=$SCRATCH/torch_compile_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"

mkdir -p "$TORCH_HOME" "$TORCHINDUCTOR_CACHE_DIR"

# ── Init ──────────────────────────────────────────────────────────────────────
OUT="$SLURM_SUBMIT_DIR/outputs_news2/experiment_logs"
CONFIG_DIR="$SLURM_SUBMIT_DIR/configs/multi/current/models_comparison"

mkdir -p "$OUT"
mkdir -p "$CONFIG_DIR"

echo "════════════════════════════════════════════════════════════════"
echo "GPU Info"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "════════════════════════════════════════════════════════════════"
echo "Model sweep — $(date)"
echo ""

# ── Models ────────────────────────────────────────────────────────────────────
MODELS=(
    "Qwen/Qwen3-32B"
)

NUM_GPUS=4

declare -A GPU_BUSY=()
declare -A PID_GPU=()
declare -A PID_LABEL=()

N_BUSY=0
FAILED=0

pick_free_gpu() {
    for g in $(seq 0 $((NUM_GPUS-1))); do
        if [[ -z "${GPU_BUSY[$g]:-}" ]]; then
            echo "$g"
            return
        fi
    done
}

reap_one() {
    local done_pid rc=0
    wait -n -p done_pid || rc=$?
    local gpu="${PID_GPU[$done_pid]}"
    local label="${PID_LABEL[$done_pid]}"

    if (( rc == 0 )); then
        echo "  OK  [$label] completed"
    else
        echo "  FAILED  [$label] (exit $rc)"
        FAILED=$((FAILED+1))
    fi

    unset 'GPU_BUSY[$gpu]'
    unset 'PID_GPU[$done_pid]'
    unset 'PID_LABEL[$done_pid]'

    N_BUSY=$((N_BUSY-1))
}

for MODEL in "${MODELS[@]}"; do

    NAME=$(basename "$MODEL")
    CONFIG="$CONFIG_DIR/${NAME}.yaml"

    cat > "$CONFIG" <<EOF
experiment:
  seed: 40
  n_jobs: 4
  output_dir: results/multi/current/models_comparison/$NAME
  horizon: 500
  runs: 20
  order: [ucb, Greedy, ts, LLM]
  track_other_actions_for: ["ucb", "Greedy", "ts"]
  add_opt: False

environment:
  n_arms: 5
  delta: 0.2
  best_mean: 0.6
  probs: [0.4, 0.4, 0.6, 0.4, 0.4]

agents:
  - name: ucb
    class: UCB

  - name: ts
    class: TS

  - name: Greedy
    class: Greedy

  - name: LLM
    class: LLM
    prompt: default
    params:
      model: $MODEL
EOF

    while (( N_BUSY >= NUM_GPUS )); do
        reap_one
    done

    gpu=$(pick_free_gpu)
    log="${OUT}/sweep_model_${NAME}_gpu${gpu}.log"
    echo "Launching: $NAME on GPU $gpu -> $log"

    CUDA_VISIBLE_DEVICES=$gpu \
    python "$SLURM_SUBMIT_DIR/experiments/run_llm.py" \
        --config "$CONFIG" \
        --resume \
        >"$log" 2>&1 &
        

    pid=$!
    GPU_BUSY[$gpu]=$pid
    PID_GPU[$pid]=$gpu
    PID_LABEL[$pid]="GPU${gpu}:${NAME}"
    N_BUSY=$((N_BUSY+1))

done

echo ""
echo "Draining remaining sweeps..."
echo "════════════════════════════════════════════════════════════════"

while (( N_BUSY > 0 )); do
    reap_one
done

echo ""
echo "════════════════════════════════════════════════════════════════"

if [[ $FAILED -eq 0 ]]; then
    echo "All model sweeps completed successfully at $(date)"
else
    echo "$FAILED sweep(s) failed at $(date)"
fi

echo "════════════════════════════════════════════════════════════════"

rm -rf "${TORCHINDUCTOR_CACHE_DIR:?}"/*
echo "Job finished at $(date)"
exit $FAILED
