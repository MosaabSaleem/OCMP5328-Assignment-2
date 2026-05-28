"""
Step 0 — Materialize the base_gemma evaluation reference + warm HF cache.

Two jobs, both idempotent:

1. Create results/models/base_gemma/ with model_reference.json + a local
   copy of the tokenizer, so _model_helpers.load_model can treat the
   untouched Gemma model with the same interface as the LoRA-adapted ones
   (steps 4b-8 iterate over MODELS_TO_EVAL which includes 'base_gemma').

2. Pre-warm the Hugging Face cache by downloading Gemma's full snapshot.
   Without this, the first eval step that touches base_gemma pays the
   ~2.5 GB download cost mid-pipeline. snapshot_download is a no-op if
   the repo is already cached.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Algorithm.config import MODEL_NAME, MODEL_DIR, HF_TOKEN

from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

base_dir = os.path.join(MODEL_DIR, "base_gemma")
os.makedirs(base_dir, exist_ok=True)

ref_path = os.path.join(base_dir, "model_reference.json")
tok_path = os.path.join(base_dir, "tokenizer.json")

if os.path.isfile(ref_path) and os.path.isfile(tok_path):
    print(f"[Step 0] base_gemma marker already set up at {base_dir}")
else:
    print(f"[Step 0] Setting up base_gemma reference at {base_dir}")
    ref = {
        "name": "base_gemma",
        "source_model_key": "base_gemma",
        "model_type": "huggingface_reference",
        "base_model": MODEL_NAME,
        "training_dataset": None,
        "adapter_path": None,
        "description": (
            "Untouched Hugging Face base model used as the pre-LoRA reference. "
            "Weights are loaded on demand from Hugging Face; only the tokenizer "
            "is stored locally so load_model() has the same interface as for "
            "the LoRA-adapted models."
        ),
    }
    with open(ref_path, "w") as f:
        json.dump(ref, f, indent=2)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    tok.save_pretrained(base_dir)
    print(f"[Step 0] Wrote model_reference.json and tokenizer files to {base_dir}")

# Pre-warm the HuggingFace cache (no-op if already present).
print(f"[Step 0] Warming HF cache for {MODEL_NAME} (no-op if cached)...")
cache_path = snapshot_download(repo_id=MODEL_NAME, token=HF_TOKEN)
print(f"[Step 0] HF cache ready at {cache_path}")
