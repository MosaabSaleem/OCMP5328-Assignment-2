"""
Step 4 — Train debiased model using CDA + Counterfactual Logit Pairing (CLP) with LoRA.

Proposed training objective:
    L = L_LM(x) + L_LM(x_cf) + lambda * L_CLP(x, x_cf)

Where:
    L_LM     = standard causal language modelling loss
    L_CLP    = symmetric KL divergence between original and counterfactual
               output logit distributions. Forces the model to give
               gender-invariant predictions.
    lambda   = LAMBDA_CLP hyperparameter weighting the fairness penalty

This builds on CDA (Zhao et al. 2018) with the CLP regularisation term
(Garg et al. 2019). LoRA keeps fine-tuning efficient.
Refs:
  CDA    : Zhao et al., 2018. https://doi.org/10.18653/v1/N18-2003
  CLP    : Garg et al., 2019. https://doi.org/10.1145/3306618.3317950
  LoRA   : Hu et al., 2022.   https://doi.org/10.48550/arXiv.2106.09685
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Algorithm.config import (MODEL_NAME, DATA_DIR, MODEL_DIR, METRICS_DIR,
                               EPOCHS, BATCH_SIZE, GRAD_ACCUM, LR, MAX_LENGTH, LAMBDA_CLP,
                               LORA_R, LORA_ALPHA, LORA_DROPOUT)

import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CDAPairDataset(Dataset):
    """Loads (original, counterfactual) biography pairs for CLP training."""
    def __init__(self, df, tokenizer, max_length):
        self.orig = df["text"].astype(str).tolist()
        self.cf   = df["text_cf"].astype(str).tolist()
        self.tok  = tokenizer
        self.maxl = max_length

    def __len__(self):
        return len(self.orig)

    def _enc(self, text):
        return self.tok(text, truncation=True, padding="max_length",
                        max_length=self.maxl, return_tensors="pt")

    def __getitem__(self, idx):
        o = self._enc(self.orig[idx])
        c = self._enc(self.cf[idx])
        return {
            "o_ids":  o["input_ids"].squeeze(0),
            "o_mask": o["attention_mask"].squeeze(0),
            "c_ids":  c["input_ids"].squeeze(0),
            "c_mask": c["attention_mask"].squeeze(0),
        }


def compute_clp_loss(logits_o, logits_c, mask):
    """
    Symmetric KL divergence between token distributions of the
    original and counterfactual sentence pair, averaged over
    non-padding positions.
    L_CLP = 0.5 * ( KL(p_o || p_c) + KL(p_c || p_o) )
    """
    p_o = F.softmax(logits_o, dim=-1).clamp(min=1e-9)
    p_c = F.softmax(logits_c, dim=-1).clamp(min=1e-9)
    kl_oc = (p_o * (p_o.log() - p_c.log())).sum(dim=-1)
    kl_co = (p_c * (p_c.log() - p_o.log())).sum(dim=-1)
    skl   = 0.5 * (kl_oc + kl_co) * mask.float()
    return skl.sum() / mask.float().sum().clamp_min(1.0)


print(f"[Step 4] Training DEBIASED model (CDA + CLP + LoRA)  lambda={LAMBDA_CLP}")

df = pd.read_csv(os.path.join(DATA_DIR, "bias_in_bios_pairs.csv")).dropna(subset=["text","text_cf"])
print(f"[Step 4] Training pairs: {len(df)}")

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
mdl = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
mdl = get_peft_model(mdl, LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
))
mdl.to(DEVICE)
mdl.print_trainable_parameters()
mdl.train()
print(f"[Step 4] Training on device: {DEVICE}")

loader  = DataLoader(CDAPairDataset(df, tok, MAX_LENGTH),
                     batch_size=BATCH_SIZE, shuffle=True)
opt     = AdamW(mdl.parameters(), lr=LR)
history = []
t0      = time.time()
opt.zero_grad()

for epoch in range(EPOCHS):
    for step, batch in enumerate(loader):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        o_labels = batch["o_ids"].clone()
        c_labels = batch["c_ids"].clone()
        o_labels[batch["o_mask"] == 0] = -100
        c_labels[batch["c_mask"] == 0] = -100

        out_o = mdl(input_ids=batch["o_ids"], attention_mask=batch["o_mask"],
                    labels=o_labels)
        out_c = mdl(input_ids=batch["c_ids"], attention_mask=batch["c_mask"],
                    labels=c_labels)
        l_clp = compute_clp_loss(out_o.logits, out_c.logits, batch["o_mask"])
        loss  = out_o.loss + out_c.loss + LAMBDA_CLP * l_clp
        scaled_loss = loss / GRAD_ACCUM

        scaled_loss.backward()
        if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(loader):
            opt.step()
            opt.zero_grad()

        history.append({
            "epoch": epoch + 1, "step": step + 1,
            "loss_total":   round(float(loss),      4),
            "loss_lm_orig": round(float(out_o.loss), 4),
            "loss_lm_cf":   round(float(out_c.loss), 4),
            "loss_clp":     round(float(l_clp),      4),
        })
        if (step + 1) % 20 == 0:
            print(history[-1])

elapsed   = round(time.time() - t0, 2)
save_path = os.path.join(MODEL_DIR, "debiased")
mdl.save_pretrained(save_path)
tok.save_pretrained(save_path)

with open(os.path.join(METRICS_DIR, "train_debiased.json"), "w") as f:
    json.dump({
        "model": "debiased", "train_seconds": elapsed,
        "train_rows": len(df), "epochs": EPOCHS,
        "grad_accum": GRAD_ACCUM,
        "lambda_clp": LAMBDA_CLP,
        "final_step": history[-1] if history else {}
    }, f, indent=2)
print(f"[Step 4] Done. Saved to {save_path}  ({elapsed}s)")
