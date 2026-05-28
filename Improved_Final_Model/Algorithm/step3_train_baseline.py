"""
Step 3 — Train LoRA baseline on original (non-augmented) Bias-in-Bios data.
This is the model we compare against in all evaluations.
LoRA freezes the base Gemma weights and trains small adapter matrices only,
making fine-tuning feasible without a large GPU.
Ref: Hu et al., 2022. https://doi.org/10.48550/arXiv.2106.09685
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Algorithm.config import (MODEL_NAME, DATA_DIR, MODEL_DIR, METRICS_DIR,
                               EPOCHS, BATCH_SIZE, GRAD_ACCUM, LR, MAX_LENGTH,
                               WARMUP_RATIO, LR_SCHEDULER,
                               LORA_R, LORA_ALPHA, LORA_DROPOUT, HF_TOKEN)

import pandas as pd
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          Trainer, TrainingArguments,
                          DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model

from Algorithm._wandb_log import (start as wandb_start,
                                   finish as wandb_finish,
                                   log_lora_artifact)

print(f"[Step 3] Training BASELINE model with LoRA on original data...")

# Load data
df = pd.read_csv(os.path.join(DATA_DIR, "bias_in_bios.csv")).dropna(subset=["text"])
print(f"[Step 3] Training rows: {len(df)}")

# Load model + tokenizer
tok = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
mdl = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=HF_TOKEN)

# Attach LoRA adapters
lora_cfg = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
mdl = get_peft_model(mdl, lora_cfg)
mdl.print_trainable_parameters()

# Tokenize
ds = Dataset.from_dict({"text": df["text"].astype(str).tolist()})
def tokenize(batch):
    enc = tok(batch["text"], truncation=True, padding="max_length",
              max_length=MAX_LENGTH)
    enc["labels"] = [
        [tok_id if mask == 1 else -100
         for tok_id, mask in zip(ids, attn)]
        for ids, attn in zip(enc["input_ids"], enc["attention_mask"])
    ]
    return enc
ds = ds.map(tokenize, batched=True, remove_columns=["text"])

# Start the W&B run before Trainer init so Trainer's report_to="wandb"
# reuses the active run instead of creating a second, anonymous one.
wb_run = wandb_start(
    job_type="train",
    name="baseline_train",
    config={
        "model": MODEL_NAME, "method": "baseline_lora",
        "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM, "lr": LR,
        "max_length": MAX_LENGTH, "warmup_ratio": WARMUP_RATIO,
        "lr_scheduler": LR_SCHEDULER,
        "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
        "train_rows": len(df),
    },
)

# Train — per-epoch checkpoints (save_total_limit=1) act as crash insurance:
# if the next epoch dies we can resume; otherwise the final save_pretrained()
# below leaves the canonical adapter at MODEL_DIR/baseline.
args = TrainingArguments(
    output_dir=os.path.join(MODEL_DIR, "baseline"),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type=LR_SCHEDULER,
    warmup_ratio=WARMUP_RATIO,
    logging_steps=10,
    report_to="wandb" if wb_run is not None else "none",
    save_strategy="epoch",
    save_total_limit=1,
    remove_unused_columns=False,
)
t0 = time.time()
Trainer(model=mdl, args=args, train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False)).train()
elapsed = round(time.time() - t0, 2)

# Save
save_path = os.path.join(MODEL_DIR, "baseline")
mdl.save_pretrained(save_path)
tok.save_pretrained(save_path)

with open(os.path.join(METRICS_DIR, "train_baseline.json"), "w") as f:
    json.dump({"model": "baseline", "train_seconds": elapsed,
               "train_rows": len(df), "epochs": EPOCHS}, f, indent=2)

if wb_run is not None:
    wb_run.summary["train_seconds"] = elapsed
    wb_run.summary["train_rows"]    = len(df)
log_lora_artifact(
    wb_run, name="baseline_lora", save_path=save_path,
    metadata={"base_model": MODEL_NAME, "method": "baseline_lora",
              "train_seconds": elapsed, "train_rows": len(df),
              "epochs": EPOCHS, "lr": LR, "max_length": MAX_LENGTH,
              "lora_r": LORA_R, "lora_alpha": LORA_ALPHA},
)
wandb_finish(wb_run)
print(f"[Step 3] Done. Saved to {save_path}  ({elapsed}s)")
