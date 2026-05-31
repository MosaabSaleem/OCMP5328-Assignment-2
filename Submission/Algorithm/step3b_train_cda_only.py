"""
Step 3b — Train LoRA on CDA augmented data without the CLP regulariser.

Skips retraining if results/models/cda_only/ already holds an adapter
(set FORCE_RETRAIN=1 to override); otherwise resumes from any local checkpoint.
"""
import sys, os, time, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Algorithm.config import (MODEL_NAME, DATA_DIR, MODEL_DIR, METRICS_DIR,
                               EPOCHS, BATCH_SIZE, GRAD_ACCUM, LR, MAX_LENGTH,
                               WARMUP_RATIO, LR_SCHEDULER,
                               LORA_R, LORA_ALPHA, LORA_DROPOUT, HF_TOKEN)

import torch
import pandas as pd
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          Trainer, TrainingArguments,
                          DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model

from Algorithm._wandb_log import (start as wandb_start,
                                   finish as wandb_finish,
                                   log_lora_artifact)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


SAVE_PATH = os.path.join(MODEL_DIR, "cda_only")


def _adapter_already_exists(path: str) -> bool:
    """True if a complete LoRA adapter (config + weights) already exists."""
    return (
        os.path.isfile(os.path.join(path, "adapter_config.json"))
        and (
            os.path.isfile(os.path.join(path, "adapter_model.safetensors"))
            or os.path.isfile(os.path.join(path, "adapter_model.bin"))
        )
    )


if _adapter_already_exists(SAVE_PATH) and os.environ.get("FORCE_RETRAIN", "0") != "1":
    print(f"[Step 3b] ✓ CDA-only adapter already exists at {SAVE_PATH}")
    print("[Step 3b] Skipping training (set FORCE_RETRAIN=1 to force a full retrain).")
    sys.exit(0)

print("[Step 3b] Training CDA-ONLY model with LoRA on CDA-augmented data...")

# Build the CDA-augmented training set: original + counterfactual rows
# stacked into a single 'text' column so the existing Trainer pipeline can
# consume it unchanged. Trainer shuffles by default so the model does not
# see all originals before all counterfactuals.
df = pd.read_csv(os.path.join(DATA_DIR, "bias_in_bios_pairs.csv")).dropna(
    subset=["text", "text_cf"]
)
texts = df["text"].astype(str).tolist() + df["text_cf"].astype(str).tolist()
print(f"[Step 3b] Training rows: {len(texts)}  ({len(df)} originals + {len(df)} counterfactuals)")

tok = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

mdl = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, token=HF_TOKEN,
    attn_implementation="sdpa",
)
mdl.config.use_cache = False

lora_cfg = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
mdl = get_peft_model(mdl, lora_cfg)
mdl.enable_input_require_grads()
mdl.print_trainable_parameters()

ds = Dataset.from_dict({"text": texts})
def tokenize(batch):
    # No padding here — the collator pads to the longest sequence
    return tok(batch["text"], truncation=True, max_length=MAX_LENGTH)
ds = ds.map(tokenize, batched=True, remove_columns=["text"])
wb_run = wandb_start(
    job_type="train",
    name="cda_only_train",
    config={
        "model": MODEL_NAME, "method": "cda_only_lora",
        "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM, "lr": LR,
        "max_length": MAX_LENGTH, "warmup_ratio": WARMUP_RATIO,
        "lr_scheduler": LR_SCHEDULER,
        "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
        "train_rows": len(texts),
        "n_originals": len(df), "n_counterfactuals": len(df),
        "precision": "fp16", "attn_impl": "sdpa",
    },
)

args = TrainingArguments(
    output_dir=SAVE_PATH,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type=LR_SCHEDULER,
    warmup_ratio=WARMUP_RATIO,
    fp16=True,
    optim="adamw_torch_fused",
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    train_sampling_strategy="group_by_length",
    logging_steps=50,
    report_to="wandb" if wb_run is not None else "none",
    save_strategy="epoch",
    save_total_limit=1,
    remove_unused_columns=False,
)

# Auto-resume from the latest local checkpoint if one exists (only when
# the final adapter is missing, which was already checked above).
resume = bool(glob.glob(os.path.join(SAVE_PATH, "checkpoint-*")))
if resume:
    print(f"[Step 3b] Found existing checkpoint in {SAVE_PATH}; resuming.")

t0 = time.time()
Trainer(model=mdl, args=args, train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(
            tok, mlm=False, pad_to_multiple_of=8
        )).train(resume_from_checkpoint=resume)
elapsed = round(time.time() - t0, 2)

mdl.save_pretrained(SAVE_PATH)
tok.save_pretrained(SAVE_PATH)

with open(os.path.join(METRICS_DIR, "train_cda_only.json"), "w") as f:
    json.dump({"model": "cda_only", "train_seconds": elapsed,
               "train_rows": len(texts), "epochs": EPOCHS}, f, indent=2)

if wb_run is not None:
    wb_run.summary["train_seconds"] = elapsed
    wb_run.summary["train_rows"]    = len(texts)
log_lora_artifact(
    wb_run, name="cda_only_lora", save_path=SAVE_PATH,
    metadata={"base_model": MODEL_NAME, "method": "cda_only_lora",
              "train_seconds": elapsed, "train_rows": len(texts),
              "epochs": EPOCHS, "lr": LR, "max_length": MAX_LENGTH,
              "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
              "precision": "fp16", "attn_impl": "sdpa"},
)
wandb_finish(wb_run)
print(f"[Step 3b] Done. Saved to {SAVE_PATH}  ({elapsed}s)")