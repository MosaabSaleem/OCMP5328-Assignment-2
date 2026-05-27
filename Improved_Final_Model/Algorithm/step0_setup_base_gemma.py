"""
Step 0 — Materialize the base_gemma evaluation reference.

The eval steps (5-8) iterate over MODELS_TO_EVAL, which includes the untouched
pre-LoRA Gemma model under the key 'base_gemma'. _model_helpers.load_model
expects results/models/base_gemma/ to exist with a model_reference.json plus a
local tokenizer, so we keep the same load interface across all four models
without duplicating ~2GB of weights to disk.

This step is idempotent: if model_reference.json + tokenizer files are already
present, nothing is rewritten.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Algorithm.config import MODEL_NAME, MODEL_DIR

from transformers import AutoTokenizer

base_dir = os.path.join(MODEL_DIR, "base_gemma")
os.makedirs(base_dir, exist_ok=True)

ref_path = os.path.join(base_dir, "model_reference.json")
tok_path = os.path.join(base_dir, "tokenizer.json")

if os.path.isfile(ref_path) and os.path.isfile(tok_path):
    print(f"[Step 0] base_gemma already set up at {base_dir} — skipping")
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

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.save_pretrained(base_dir)
    print(f"[Step 0] Wrote model_reference.json and tokenizer files to {base_dir}")
