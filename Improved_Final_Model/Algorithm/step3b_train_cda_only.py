"""
Step 3b — Train LoRA on CDA-augmented data WITHOUT the CLP regulariser.
This is the ablation arm that isolates the contribution of CDA from CLP:
  baseline  : LoRA on raw Bias-in-Bios               (no CDA, no CLP)
  cda_only  : LoRA on CDA-augmented Bias-in-Bios     (CDA only)
  debiased  : LoRA on CDA pairs + CLP penalty        (CDA + CLP)
Concretely, we train on the original biographies and their gender-swapped
counterfactuals concatenated as plain LM examples — same trainer, same
hyperparameters as step 3, just twice as many rows from CDA.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Algorithm.config import (MODEL_NAME, DATA_DIR, MODEL_DIR, METRICS_DIR,
                               EPOCHS, BATCH_SIZE, GRAD_ACCUM, LR, MAX_LENGTH,
                               WARMUP_RATIO, LR_SCHEDULER,
                               LORA_R, LORA_ALPHA, LORA_DROPOUT)

import pandas as pd
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          Trainer, TrainingArguments,
                          DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model

from Algorithm._wandb_log import enabled as wandb_enabled, ensure_group

print(f"[Step 3b] Training CDA-ONLY model with LoRA on CDA-augmented data...")

USE_WANDB = wandb_enabled()
if USE_WANDB:
    ensure_group()
    os.environ["WANDB_NAME"] = "cda_only_train"

# Build the CDA-augmented training set: original + counterfactual rows
# stacked into a single 'text' column so the existing Trainer pipeline can
# consume it unchanged. Trainer shuffles by default so the model does not
# see all originals before all counterfactuals.
df = pd.read_csv(os.path.join(DATA_DIR, "bias_in_bios_pairs.csv")).dropna(
    subset=["text", "text_cf"]
)
texts = df["text"].astype(str).tolist() + df["text_cf"].astype(str).tolist()
print(f"[Step 3b] Training rows: {len(texts)}  ({len(df)} originals + {len(df)} counterfactuals)")

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
mdl = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

lora_cfg = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
mdl = get_peft_model(mdl, lora_cfg)
mdl.print_trainable_parameters()

ds = Dataset.from_dict({"text": texts})
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

args = TrainingArguments(
    output_dir=os.path.join(MODEL_DIR, "cda_only"),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type=LR_SCHEDULER,
    warmup_ratio=WARMUP_RATIO,
    logging_steps=10,
    report_to="wandb" if USE_WANDB else "none",
    save_strategy="no",
    remove_unused_columns=False,
)
t0 = time.time()
Trainer(model=mdl, args=args, train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False)).train()
elapsed = round(time.time() - t0, 2)

save_path = os.path.join(MODEL_DIR, "cda_only")
mdl.save_pretrained(save_path)
tok.save_pretrained(save_path)

with open(os.path.join(METRICS_DIR, "train_cda_only.json"), "w") as f:
    json.dump({"model": "cda_only", "train_seconds": elapsed,
               "train_rows": len(texts), "epochs": EPOCHS}, f, indent=2)
print(f"[Step 3b] Done. Saved to {save_path}  ({elapsed}s)")
