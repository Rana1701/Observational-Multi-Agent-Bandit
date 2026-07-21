#!/usr/bin/env bash
# Pre-download the LLM judges from the HuggingFace Hub into a shared cache.
#
# IMPORTANT: run this on a LOGIN node (which has internet) — NOT via sbatch.
# Compute nodes on Alliance Canada / Compute Canada clusters have no outbound
# network, so models must be fetched ahead of time. Once cached, batch jobs
# load them offline (HF_HUB_OFFLINE=1).
#
# WHERE TO STORE: use the PROJECT space, not scratch. Per the Alliance Canada
# storage guide, /project is backed up, not purged, and shared with your
# research group — the right home for model weights several students reuse.
# Scratch is 20 TB but PURGED after 60 days and per-user, so shared/kept models
# should NOT live there. This script therefore targets the project space by
# default and applies group-read permissions at the end.
#
# Usage (login node):
#   HF_HOME="$HOME/projects/def-<pi>/shared/hf_cache" bash scripts/download_llms.sh
#   # gated repos (Ministral / Mistral) also need an accepted licence + token:
#   HF_TOKEN=hf_xxx HF_HOME=".../shared/hf_cache" bash scripts/download_llms.sh
#
# Then, in your jobs (any group member):
#   export HF_HOME="$HOME/projects/def-<pi>/shared/hf_cache"
#   export HF_HUB_OFFLINE=1
#   # load models by their repo id, e.g. "Qwen/Qwen2.5-7B" — resolved from cache.
#   # For best I/O on repeated runs, optionally copy to $SLURM_TMPDIR at job start.
set -euo pipefail

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
# Prefer the project space (shared, backed up, not purged). Fall back to scratch
# with a warning only if no project directory can be located.
if [[ -z "${HF_HOME:-}" ]]; then
    proj="$(ls -d "$HOME"/projects/*/ 2>/dev/null | head -1)"
    if [[ -n "$proj" ]]; then
        HF_HOME="${proj%/}/shared/hf_cache"
    else
        HF_HOME="${SCRATCH:-$HOME}/hf_cache"
        echo "WARNING: no ~/projects/<group> found; falling back to $HF_HOME."
        echo "         scratch is PURGED after 60 days and is per-user —"
        echo "         set HF_HOME to your project space to keep/share the models."
    fi
fi
export HF_HOME
mkdir -p "$HF_HOME"

# The seven LLM judges used in the paper. Override by exporting MODELS=(...).
MODELS=(${MODELS[@]:-
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
})

# --------------------------------------------------------------------------- #
# Python environment (Alliance Canada). Adjust the module version as needed.  #
# --------------------------------------------------------------------------- #
if command -v module >/dev/null 2>&1; then
    module load python/3.11 >/dev/null 2>&1 || true
fi

# The download CLI is named `hf` in recent huggingface_hub, `huggingface-cli`
# before. Fall back to a throwaway venv on the login node if neither is found.
HF_CLI=""
if command -v hf >/dev/null 2>&1; then
    HF_CLI="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_CLI="huggingface-cli"
else
    echo "huggingface CLI not found — creating a temporary venv..."
    python -m venv "$HF_HOME/.dl_env"
    # shellcheck disable=SC1091
    source "$HF_HOME/.dl_env/bin/activate"
    pip install --quiet --upgrade "huggingface_hub[cli]" hf_transfer
    HF_CLI="$(command -v hf || command -v huggingface-cli)"
fi

# Faster multi-threaded downloads if hf_transfer is available (optional).
if python -c "import hf_transfer" >/dev/null 2>&1; then
    export HF_HUB_ENABLE_HF_TRANSFER=1
fi

# Authenticate for gated repos, only if a token was provided.
if [[ -n "${HF_TOKEN:-}" ]]; then
    "$HF_CLI" login --token "$HF_TOKEN" >/dev/null 2>&1 || true
fi

# --------------------------------------------------------------------------- #
# Download                                                                    #
# --------------------------------------------------------------------------- #
echo "Cache: $HF_HOME"
echo "CLI  : $HF_CLI"
echo "Models: ${#MODELS[@]}"
echo

failed=()
for repo in "${MODELS[@]}"; do
    echo "=== $repo ==="
    n=0
    # Prefer safetensors; skip the original .pth consolidated weights.
    until "$HF_CLI" download "$repo" --exclude "*.pth" "original/*"; do
        n=$((n + 1))
        if [[ $n -ge 3 ]]; then
            echo "!! FAILED after 3 attempts: $repo"
            failed+=("$repo")
            break
        fi
        echo "   retry $n/3 in 15s..."
        sleep 15
    done
    echo
done

# --------------------------------------------------------------------------- #
# Share with the group                                                        #
# --------------------------------------------------------------------------- #
# Make the cache readable and traversable by other members of the project
# allocation: g+rX = read files + enter directories; setgid on directories so
# new files inherit the project group. If the cache is on /project this lets
# any group member load the models offline.
echo "Applying group-read permissions to $HF_HOME ..."
chmod -R g+rX "$HF_HOME" 2>/dev/null || true
find "$HF_HOME" -type d -exec chmod g+s {} + 2>/dev/null || true

echo
echo "Done. Models cached under: $HF_HOME"
if [[ ${#failed[@]} -gt 0 ]]; then
    echo "The following repos failed (gated? need HF_TOKEN + accepted licence):"
    printf '  - %s\n' "${failed[@]}"
    exit 1
fi
echo "Group members can now use them with:"
echo "    export HF_HOME=\"$HF_HOME\"; export HF_HUB_OFFLINE=1"
echo "If group members still cannot read them, set the project group explicitly:"
echo "    chgrp -R def-<pi> \"$HF_HOME\"   # replace def-<pi> with your allocation group"
