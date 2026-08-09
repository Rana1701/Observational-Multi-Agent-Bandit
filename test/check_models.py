import os
from pathlib import Path

MODELS = [
    "allenai/Olmo-3-32B-Think-SFT",
    "allenai/Olmo-3-32B-Think-DPO",
    "allenai/Olmo-3-32B-Think",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-VL-32B-Thinking",
    "Qwen/Qwen2.5-32B",
    "google/gemma-3-27b-it",
    "google/gemma-2-27b-it",
    "allenai/OLMo-2-0325-32B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
]


CACHE = Path("/project/6102313/shared/hf_cache/hub")


def check(model):
    name = "models--" + model.replace("/", "--")
    folder = CACHE / name

    print("="*80)
    print(model)

    if not folder.exists():
        print("❌ absent")
        return

    snapshots = list((folder/"snapshots").glob("*"))

    if not snapshots:
        print("❌ pas de snapshot")
        return

    snap = snapshots[0]

    print("Snapshot:", snap)

    weights = list(snap.glob("*.safetensors"))

    if weights:
        size = sum(x.stat().st_size for x in weights)/1024**3
        print(f"✅ safetensors: {len(weights)} fichiers")
        print(f"   taille: {size:.2f} GB")
    else:
        print("❌ aucun safetensors")

    print("config:",
          "✅" if (snap/"config.json").exists() else "❌")

    print("tokenizer:",
          "✅" if (snap/"tokenizer.json").exists() else "❌")


for m in MODELS:
    check(m)
