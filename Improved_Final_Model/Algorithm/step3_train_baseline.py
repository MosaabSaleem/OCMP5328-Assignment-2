"""
Step 3 — Train LoRA baseline on original (non-augmented) Bias-in-Bios data.
This is the model we compare against in all evaluations.
LoRA freezes the base Gemma weights and trains small adapter matrices only,
making fine-tuning feasible without a large GPU.
"""
import sys
import os
import time
import json

from huggingface_hub import login

login("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")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Algorithm.config import (
    MODEL_NAME,
    DATA_DIR,
    MODEL_DIR,
    METRICS_DIR,
    EPOCHS,
    BATCH_SIZE,
    GRAD_ACCUM,
    LR,
    MAX_LENGTH,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
)

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

print("[Step 3] Training BASELINE model with LoRA on original data...")

df = pd.read_csv(os.path.join(DATA_DIR, "bias_in_bios.csv")).dropna(subset=["text"])
print(f"[Step 3] Training rows: {len(df)}")

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

mdl = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

lora_cfg = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
mdl = get_peft_model(mdl, lora_cfg)
mdl.print_trainable_parameters()

ds = Dataset.from_dict({"text": df["text"].astype(str).tolist()})


def tokenize(batch):
    enc = tok(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )
    enc["labels"] = enc["input_ids"].copy()
    return enc


ds = ds.map(tokenize, batched=True, remove_columns=["text"])

args = TrainingArguments(
    output_dir=os.path.join(MODEL_DIR, "baseline"),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    logging_steps=5,
    report_to="none",
    save_strategy="epoch",
    remove_unused_columns=False,
    dataloader_pin_memory=False,
)

t0 = time.time()
trainer = Trainer(
    model=mdl,
    args=args,
    train_dataset=ds,
    data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
)
trainer.train()
elapsed = round(time.time() - t0, 2)

save_path = os.path.join(MODEL_DIR, "baseline")
mdl.save_pretrained(save_path)
tok.save_pretrained(save_path)

with open(os.path.join(METRICS_DIR, "train_baseline.json"), "w") as f:
    json.dump(
        {
            "model": "baseline",
            "train_seconds": elapsed,
            "train_rows": len(df),
            "epochs": EPOCHS,
        },
        f,
        indent=2,
    )

print(f"[Step 3] Done. Saved to {save_path} ({elapsed}s)")