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
GRAD_ACCUM   = int(os.environ.get("GRAD_ACCUM",   "8"))
LR           = float(os.environ.get("LR",         "2e-4"))
MAX_LENGTH   = int(os.environ.get("MAX_LENGTH",   "256"))
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", "0.1"))
LR_SCHEDULER = os.environ.get("LR_SCHEDULER",    "cosine")
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

# ── Bias-in-Bios profession label mapping (De-Arteaga et al., 2019) ───────────
PROFESSION_LABELS = {
    0: "accountant", 1: "architect", 2: "attorney", 3: "chiropractor",
    4: "comedian", 5: "composer", 6: "dentist", 7: "dietitian",
    8: "dj", 9: "filmmaker", 10: "interior_designer", 11: "journalist",
    12: "model", 13: "nurse", 14: "painter", 15: "paralegal",
    16: "pastor", 17: "personal_trainer", 18: "photographer", 19: "physician",
    20: "poet", 21: "professor", 22: "psychologist", 23: "rapper",
    24: "software_engineer", 25: "surgeon", 26: "teacher", 27: "yoga_teacher",
}

# ── Models that every evaluation step iterates over ────────────────────────────
# base_gemma  : untouched Gemma-3-1b-pt, no fine-tuning at all
# baseline    : LoRA on raw Bias-in-Bios (control for the effect of fine-tuning)
# cda_only    : LoRA on CDA-augmented Bias-in-Bios (isolates CDA from CLP)
# debiased    : LoRA on CDA pairs + CLP (proposed method)
MODELS_TO_EVAL = ["base_gemma", "baseline", "cda_only", "debiased"]

for _d in [DATA_DIR, RESULTS_DIR, MODEL_DIR, METRICS_DIR, FIGURES_DIR,
           os.path.join(MODEL_DIR, "base_gemma"),
           os.path.join(MODEL_DIR, "baseline"),
           os.path.join(MODEL_DIR, "cda_only"),
           os.path.join(MODEL_DIR, "debiased")]:
    os.makedirs(_d, exist_ok=True)
