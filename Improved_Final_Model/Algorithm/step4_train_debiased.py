"""
Step 4 — Train debiased model using CDA + Counterfactual Logit Pairing (CLP) with LoRA.

Proposed training objective:
L = L_LM(x) + L_LM(x_cf) + lambda * L_CLP(x, x_cf)
"""
import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Algorithm.config import (
    MODEL_NAME,
    DATA_DIR,
    MODEL_DIR,
    METRICS_DIR,
    EPOCHS,
    BATCH_SIZE,
    LR,
    MAX_LENGTH,
    LAMBDA_CLP,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
)

import torch
import torch.nn.functional as F
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CDAPairDataset(Dataset):
    """Loads (original, counterfactual) biography pairs for CLP training."""

    def __init__(self, df, tokenizer, max_length):
        self.orig = df["text"].astype(str).tolist()
        self.cf = df["text_cf"].astype(str).tolist()
        self.tok = tokenizer
        self.maxl = max_length

    def __len__(self):
        return len(self.orig)

    def _enc(self, text):
        return self.tok(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.maxl,
            return_tensors="pt",
        )

    def __getitem__(self, idx):
        o = self._enc(self.orig[idx])
        c = self._enc(self.cf[idx])
        return {
            "o_ids": o["input_ids"].squeeze(0),
            "o_mask": o["attention_mask"].squeeze(0),
            "c_ids": c["input_ids"].squeeze(0),
            "c_mask": c["attention_mask"].squeeze(0),
        }


def compute_clp_loss(logits_o, logits_c, mask):
    p_o = F.softmax(logits_o, dim=-1).clamp(min=1e-9)
    p_c = F.softmax(logits_c, dim=-1).clamp(min=1e-9)
    kl_oc = (p_o * (p_o.log() - p_c.log())).sum(dim=-1)
    kl_co = (p_c * (p_c.log() - p_o.log())).sum(dim=-1)
    skl = 0.5 * (kl_oc + kl_co) * mask.float()
    return skl.sum() / mask.float().sum().clamp_min(1.0)


print(f"[Step 4] Training DEBIASED model (CDA + CLP + LoRA) lambda={LAMBDA_CLP}")

df = pd.read_csv(os.path.join(DATA_DIR, "bias_in_bios_pairs.csv")).dropna(subset=["text", "text_cf"])
print(f"[Step 4] Training pairs: {len(df)}")

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

mdl = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
mdl = get_peft_model(
    mdl,
    LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    ),
)
mdl.train()
mdl.to(DEVICE)
mdl.print_trainable_parameters()

loader = DataLoader(CDAPairDataset(df, tok, MAX_LENGTH), batch_size=BATCH_SIZE, shuffle=True)
opt = AdamW(mdl.parameters(), lr=LR)

history = []
t0 = time.time()

for epoch in range(EPOCHS):
    for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch + 1}")):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        opt.zero_grad()

        out_o = mdl(
            input_ids=batch["o_ids"],
            attention_mask=batch["o_mask"],
            labels=batch["o_ids"],
        )
        out_c = mdl(
            input_ids=batch["c_ids"],
            attention_mask=batch["c_mask"],
            labels=batch["c_ids"],
        )

        l_clp = compute_clp_loss(out_o.logits, out_c.logits, batch["o_mask"])
        loss = out_o.loss + out_c.loss + LAMBDA_CLP * l_clp

        loss.backward()
        opt.step()

        history.append(
            {
                "epoch": epoch + 1,
                "step": step + 1,
                "loss_total": float(loss.detach().cpu()),
                "loss_lm_orig": float(out_o.loss.detach().cpu()),
                "loss_lm_cf": float(out_c.loss.detach().cpu()),
                "loss_clp": float(l_clp.detach().cpu()),
            }
        )

elapsed = round(time.time() - t0, 2)

save_path = os.path.join(MODEL_DIR, "debiased")
mdl.save_pretrained(save_path)
tok.save_pretrained(save_path)

hist_df = pd.DataFrame(history)
hist_df.to_csv(os.path.join(METRICS_DIR, "debiased_training_history.csv"), index=False)

with open(os.path.join(METRICS_DIR, "train_debiased.json"), "w") as f:
    json.dump(
        {
            "model": "debiased",
            "train_seconds": elapsed,
            "train_rows": len(df),
            "epochs": EPOCHS,
            "lambda_clp": LAMBDA_CLP,
            "final_step": history[-1] if history else {},
        },
        f,
        indent=2,
    )

print(f"[Step 4] Done. Saved to {save_path} ({elapsed}s)")