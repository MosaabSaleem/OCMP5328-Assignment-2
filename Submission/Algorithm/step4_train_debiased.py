"""
Step 4 — Train debiased model using CDA + CLP with LoRA

Proposed training objective:
    L = L_LM(x) + L_LM(x_cf) + lambda * L_CLP(x, x_cf)

Where:
    L_LM     = standard causal language modelling loss
    L_CLP    = symmetric KL divergence between original and counterfactual
               output logit distributions. Forces the model to give
               gender invariant predictions.
    lambda   = LAMBDA_CLP hyperparameter weighting the fairness penalty

Skips retraining if results/models/debiased/ already holds an adapter
(set FORCE_RETRAIN=1 to override).
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model
from Algorithm._wandb_log import (
    start as wandb_start,
    finish as wandb_finish,
    log_lora_artifact,
)
from Algorithm.config import (
    DATA_DIR, EPOCHS, HF_TOKEN, LAMBDA_CLP,
    LORA_ALPHA, LORA_DROPOUT, LORA_R, LR, LR_SCHEDULER,
    MAX_LENGTH, METRICS_DIR, MODEL_DIR, MODEL_NAME, WARMUP_RATIO,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Override config's BATCH_SIZE/GRAD_ACCUM locally - see method notes
BATCH_SIZE = 2
GRAD_ACCUM = 4

SAVE_PATH = os.path.join(MODEL_DIR, "debiased")


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
    print(f"[Step 4] ✓ Debiased adapter already exists at {SAVE_PATH}")
    print("[Step 4] Skipping training (set FORCE_RETRAIN=1 to force a full retrain).")
    sys.exit(0)


#Load the debiasing model and tokenizer, and prepare the dataset of CDA pairs for training
class CDAPairDataset(Dataset):
    """Loads (original, counterfactual) biography pairs for CLP training.
    Tokenises without padding; the collate_fn pads dynamically per-batch."""

    def __init__(self, df, tokenizer, max_length):
        self.orig = df["text"].astype(str).tolist()
        self.cf = df["text_cf"].astype(str).tolist()
        self.tok = tokenizer
        self.maxl = max_length

    def __len__(self):
        return len(self.orig)

    def __getitem__(self, idx):
        o = self.tok(self.orig[idx], truncation=True, max_length=self.maxl)
        c = self.tok(self.cf[idx], truncation=True, max_length=self.maxl)
        return {
            "o_ids": o["input_ids"],
            "o_mask": o["attention_mask"],
            "c_ids": c["input_ids"],
            "c_mask": c["attention_mask"],
        }


def make_collate(pad_id):
    """Pad o_ids/c_ids to the longest sequence in the batch. Both o and c are padded to
    the same length so indices align for the CLP mask `o_mask[:,1:] * c_mask[:,1:]`."""

    def collate(batch):
        L = max(
            max(len(b["o_ids"]) for b in batch),
            max(len(b["c_ids"]) for b in batch),
        )
        L = ((L + 7) // 8) * 8

        def pad(seqs, val):
            return torch.tensor(
                [s + [val] * (L - len(s)) for s in seqs], dtype=torch.long
            )

        return {
            "o_ids": pad([b["o_ids"] for b in batch], pad_id),
            "o_mask": pad([b["o_mask"] for b in batch], 0),
            "c_ids": pad([b["c_ids"] for b in batch], pad_id),
            "c_mask": pad([b["c_mask"] for b in batch], 0),
        }

    return collate


def compute_clp_loss(logits_o, logits_c, mask):
    """
    Symmetric KL divergence between next token distributions of the
    original and counterfactual sentence pair, averaged over positions
    where both sequences have real tokens.

        L_CLP = 0.5 * ( KL(p_o || p_c) + KL(p_c || p_o) )

    Uses log_softmax directly for numerical stability — softmax followed by
    clamp(1e-9).log() distorts ~75% of Gemma's 262k-vocab positions per row
    by up to 27 log units, which corrupts the gradient.
    """
    # F.log_softmax computes in fp32 internally even on fp16 input — safe to
    # call inside autocast. exp() back to fp16 can underflow extreme negative
    # values to 0, which is mathematically correct (they contribute ~0 to KL).
    lp_o = F.log_softmax(logits_o, dim=-1)
    lp_c = F.log_softmax(logits_c, dim=-1)
    p_o, p_c = lp_o.exp(), lp_c.exp()
    kl_oc = (p_o * (lp_o - lp_c)).sum(dim=-1)
    kl_co = (p_c * (lp_c - lp_o)).sum(dim=-1)
    skl = 0.5 * (kl_oc + kl_co) * mask.float()
    return skl.sum() / mask.float().sum().clamp_min(1.0)


print(f"[Step 4] Training DEBIASED model (CDA + CLP + LoRA) lambda={LAMBDA_CLP}")

df = pd.read_csv(os.path.join(DATA_DIR, "bias_in_bios_pairs.csv")).dropna(
    subset=["text", "text_cf"]
)
print(f"[Step 4] Training pairs: {len(df)}")

tok = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# fp32 weights + fp16 autocast - Gemma overflows when its
# weights are stored in fp16, so we keep weights in fp32 and only let the
# compute run in fp16 via torch.amp.autocast.
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    attn_implementation="sdpa",
)
mdl.config.use_cache = False
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
# Required for PEFT under fp16 autocast — without this, gradients don't
# reach the LoRA adapters because the base model is frozen.
mdl.enable_input_require_grads()
# Gradient checkpointing trades ~30% extra forward compute for roughly
# halved activation memory. We only enable it here (not in step3 / 3b)
# because step4 needs to hold activations from TWO forward passes plus
# fp32 CLP softmax intermediates simultaneously.
mdl.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
mdl.to(DEVICE)
mdl.print_trainable_parameters()
mdl.train()
print(f"[Step 4] Training on device: {DEVICE}")

loader = DataLoader(
    CDAPairDataset(df, tok, MAX_LENGTH),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    collate_fn=make_collate(tok.pad_token_id),
    persistent_workers=True,
)
opt = AdamW(mdl.parameters(), lr=LR, fused=True)
total_steps = (len(loader) * EPOCHS + GRAD_ACCUM - 1) // GRAD_ACCUM
warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
scheduler = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)
scaler = torch.amp.GradScaler("cuda")
history = []
t0 = time.time()
opt.zero_grad()

wb_run = wandb_start(
    job_type="train",
    name="debiased_train",
    config={
        "model": MODEL_NAME,
        "method": "debiased_lora_cda_clp",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "lr": LR,
        "max_length": MAX_LENGTH,
        "warmup_ratio": WARMUP_RATIO,
        "lr_scheduler": LR_SCHEDULER,
        "lambda_clp": LAMBDA_CLP,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "train_rows": len(df),
        "precision": "fp16-autocast",
        "attn_impl": "sdpa",
    },
)

LOG_EVERY = 10

global_step = 0
for epoch in range(EPOCHS):
    for step, batch in enumerate(loader):
        batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
        o_labels = batch["o_ids"].clone()
        c_labels = batch["c_ids"].clone()
        o_labels[batch["o_mask"] == 0] = -100
        c_labels[batch["c_mask"] == 0] = -100

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            out_o = mdl(
                input_ids=batch["o_ids"],
                attention_mask=batch["o_mask"],
                labels=o_labels,
            )
            out_c = mdl(
                input_ids=batch["c_ids"],
                attention_mask=batch["c_mask"],
                labels=c_labels,
            )
            # logits[t] predicts token at position t+1, so shift logits left
            # by one and the mask right by one. Mask = 1 only where the
            # target position is a real token in BOTH sequences.
            shift_o_logits = out_o.logits[:, :-1, :]
            shift_c_logits = out_c.logits[:, :-1, :]
            shift_mask = batch["o_mask"][:, 1:] * batch["c_mask"][:, 1:]
            l_clp = compute_clp_loss(shift_o_logits, shift_c_logits, shift_mask)
            loss = out_o.loss + out_c.loss + LAMBDA_CLP * l_clp
            scaled_loss = loss / GRAD_ACCUM

        scaler.scale(scaled_loss).backward()
        if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(loader):
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            opt.zero_grad()
            global_step += 1

        entry = {
            "epoch": epoch + 1,
            "step": step + 1,
            "loss_total": round(float(loss), 4),
            "loss_lm_orig": round(float(out_o.loss), 4),
            "loss_lm_cf": round(float(out_c.loss), 4),
            "loss_clp": round(float(l_clp), 4),
        }
        history.append(entry)

        if (step + 1) % LOG_EVERY == 0:
            print(entry)
            if wb_run is not None:
                wb_run.log(
                    {
                        "train/loss_total": entry["loss_total"],
                        "train/loss_lm_orig": entry["loss_lm_orig"],
                        "train/loss_lm_cf": entry["loss_lm_cf"],
                        "train/loss_clp": entry["loss_clp"],
                        "train/epoch": epoch + 1,
                        "train/global_step": global_step,
                    }
                )

    # End-of-epoch checkpoint manually due to custom loop
    mdl.save_pretrained(SAVE_PATH)
    tok.save_pretrained(SAVE_PATH)
    print(f"[Step 4] Epoch {epoch + 1}/{EPOCHS} checkpoint saved to {SAVE_PATH}")

elapsed = round(time.time() - t0, 2)
mdl.save_pretrained(SAVE_PATH)
tok.save_pretrained(SAVE_PATH)

hist_df = pd.DataFrame(history)
hist_df.to_csv(os.path.join(METRICS_DIR, "debiased_training_history.csv"), index=False)
#Save training metrics to a json file for later analysis
with open(os.path.join(METRICS_DIR, "train_debiased.json"), "w") as f:
    json.dump(
        {
            "model": "debiased",
            "train_seconds": elapsed,
            "train_rows": len(df),
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "lambda_clp": LAMBDA_CLP,
            "final_step": history[-1] if history else {},
        },
        f,
        indent=2,
    )

# Persist the full loss curve, need to do explicitly in this custom loop
with open(os.path.join(METRICS_DIR, "train_history_debiased.json"), "w") as f:
    json.dump(history, f, indent=2)
if wb_run is not None:
    wb_run.summary["train_seconds"] = elapsed
    wb_run.summary["final_loss_total"] = history[-1]["loss_total"] if history else None
    wb_run.summary["final_loss_clp"] = history[-1]["loss_clp"] if history else None
log_lora_artifact(
    wb_run,
    name="debiased_lora",
    save_path=SAVE_PATH,
    metadata={
        "base_model": MODEL_NAME,
        "method": "debiased_lora_cda_clp",
        "train_seconds": elapsed,
        "train_rows": len(df),
        "epochs": EPOCHS,
        "lr": LR,
        "max_length": MAX_LENGTH,
        "lambda_clp": LAMBDA_CLP,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "precision": "fp16-autocast",
        "attn_impl": "sdpa",
        "final_loss_total": history[-1]["loss_total"] if history else None,
        "final_loss_clp": history[-1]["loss_clp"] if history else None,
    },
)
wandb_finish(wb_run)
print(f"[Step 4] Done. Saved to {SAVE_PATH}  ({elapsed}s)")
