"""
config.py — edit this file to change model, dataset sizes, and paths.
All other scripts import from here.
"""
import os

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_NAME = os.environ.get("MODEL_NAME", "google/gemma-3-1b-pt")

# ── Dataset sizes ──────────────────────────────────────────────────────────────
SAMPLE_SIZE = int(os.environ.get("SAMPLE_SIZE", "500"))   # Bias-in-Bios rows
EVAL_SIZE = int(os.environ.get("EVAL_SIZE", "50"))        # rows per eval dataset
SEED = int(os.environ.get("SEED", "42"))

# ── Training hyperparameters ───────────────────────────────────────────────────
EPOCHS = int(os.environ.get("EPOCHS", "2"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "8"))
LR = float(os.environ.get("LR", "8e-4"))
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "128"))
LAMBDA_CLP = float(os.environ.get("LAMBDA_CLP", "1.5"))   # weight for CLP loss

# ── LoRA settings ──────────────────────────────────────────────────────────────
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# ── Paths (auto-created) ───────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "Input_data")
RESULTS_DIR = os.path.join(ROOT, "results")
MODEL_DIR = os.path.join(RESULTS_DIR, "models")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

for _d in [
    DATA_DIR,
    RESULTS_DIR,
    MODEL_DIR,
    METRICS_DIR,
    FIGURES_DIR,
    os.path.join(MODEL_DIR, "baseline"),
    os.path.join(MODEL_DIR, "debiased"),
]:
    os.makedirs(_d, exist_ok=True)