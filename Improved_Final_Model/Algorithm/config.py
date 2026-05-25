"""
config.py — edit this file to change model, dataset sizes, and paths.
All other scripts import from here.
"""
import os

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_NAME   = os.environ.get("MODEL_NAME",   "google/gemma-3-1b-pt")

# ── Dataset sizes (lower for CPU / smoke tests) ────────────────────────────────
TRAIN_SAMPLE_SIZE = int(os.environ.get(
    "TRAIN_SAMPLE_SIZE", os.environ.get("SAMPLE_SIZE", "1000")
))  # Bias-in-Bios training rows
EVAL_SAMPLE_SIZE = int(os.environ.get(
    "EVAL_SAMPLE_SIZE", os.environ.get("EVAL_SIZE", "500")
))  # upper bound per eval dataset (most gender subsets are smaller than this)
SAMPLE_SIZE = TRAIN_SAMPLE_SIZE  # backwards-compatible alias
EVAL_SIZE = EVAL_SAMPLE_SIZE     # backwards-compatible alias
SEED         = int(os.environ.get("SEED",         "42"))

# ── Training hyperparameters ───────────────────────────────────────────────────
EPOCHS       = int(os.environ.get("EPOCHS",       "2"))
BATCH_SIZE   = int(os.environ.get("BATCH_SIZE",   "1"))
GRAD_ACCUM   = int(os.environ.get("GRAD_ACCUM",   "2"))
LR           = float(os.environ.get("LR",         "2e-4"))
MAX_LENGTH   = int(os.environ.get("MAX_LENGTH",   "128"))
LAMBDA_CLP   = float(os.environ.get("LAMBDA_CLP", "3.0"))  # weight for CLP loss

# ── LoRA settings ──────────────────────────────────────────────────────────────
LORA_R       = 8
LORA_ALPHA   = 16
LORA_DROPOUT = 0.05

# ── Paths (auto-created) ───────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(ROOT, "Input_data")
RESULTS_DIR = os.path.join(ROOT, "results")
MODEL_DIR   = os.path.join(RESULTS_DIR, "models")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# ── Models that every evaluation step iterates over ────────────────────────────
# base_gemma  : untouched Gemma-3-1b-pt, no fine-tuning at all
# baseline    : LoRA on raw Bias-in-Bios (control for the effect of fine-tuning)
# debiased    : LoRA on CDA pairs + CLP (proposed method)
MODELS_TO_EVAL = ["base_gemma", "baseline", "debiased"]

for _d in [DATA_DIR, RESULTS_DIR, MODEL_DIR, METRICS_DIR, FIGURES_DIR,
           os.path.join(MODEL_DIR, "base_gemma"),
           os.path.join(MODEL_DIR, "baseline"),
           os.path.join(MODEL_DIR, "debiased")]:
    os.makedirs(_d, exist_ok=True)
