#!/bin/bash

MODELS=(
    "allenai/Olmo-3-32B-Think-SFT"
    "allenai/Olmo-3-32B-Think-DPO"
    "allenai/Olmo-3-32B-Think"
    "Qwen/Qwen3-32B"
    "Qwen/Qwen3-VL-32B-Thinking"
    "Qwen/Qwen2.5-32B"
    "google/gemma-3-27b-it"
    "google/gemma-2-27b-it"
    "allenai/OLMo-2-0325-32B-Instruct"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
)

CONFIG_DIR="configs/multi/current/models_comparison"

for MODEL in "${MODELS[@]}"; do
    NAME=$(basename "$MODEL")
    CONFIG="$CONFIG_DIR/${NAME}.yaml"

    echo "======================================"
    echo "Running $MODEL"
    echo "Configuration : $CONFIG"
    echo "======================================"

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

    python experiments/run_llm.py --config "$CONFIG"

    if [ $? -ne 0 ]; then
        echo "❌ Échec avec $MODEL"
    else
        echo "✅ Terminé : $MODEL"
    fi
done

echo "======================================"
echo "Toutes les expériences sont terminées."
echo "======================================"