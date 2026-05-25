"""
Shared model utilities used by all evaluation scripts.
Loads a (possibly LoRA-adapted) causal LM, computes sequence log-probability,
extracts last-hidden-state sentence embeddings, and runs generation.
"""
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Algorithm.config import MODEL_NAME, MAX_LENGTH

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(path):
    """
    Load a fine-tuned model from `path`. If a PEFT/LoRA adapter is present
    (adapter_config.json), load the base model and attach the adapter.
    Otherwise load `path` directly as a full model.
    """
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    adapter_cfg = os.path.join(path, "adapter_config.json")
    if os.path.isfile(adapter_cfg):
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        mdl = PeftModel.from_pretrained(base, path)
    else:
        mdl = AutoModelForCausalLM.from_pretrained(path)

    mdl.to(DEVICE)
    mdl.eval()
    return mdl, tok


@torch.no_grad()
def seq_logprob_stats(mdl, tok, text, max_length=None):
    """
    Return summed and per-token average sequence log-probability.
    We keep both because summed log-probability is the standard sequence score
    but is length-sensitive, while token-average log-probability is easier to
    compare when paired benchmark sentences have different token lengths.
    """
    max_length = max_length or MAX_LENGTH
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length).to(DEVICE)
    input_ids = enc["input_ids"]
    if input_ids.shape[1] < 2:
        return {"sum": 0.0, "avg": 0.0, "token_count": 0}
    logits = mdl(**enc).logits
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_lp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    lp_sum = float(token_lp.sum().item())
    token_count = int(token_lp.numel())
    return {
        "sum": lp_sum,
        "avg": lp_sum / max(token_count, 1),
        "token_count": token_count,
    }


@torch.no_grad()
def seq_logprob(mdl, tok, text, max_length=None, average=False):
    """Backward-compatible sequence score helper: sum by default, average if requested."""
    stats = seq_logprob_stats(mdl, tok, text, max_length=max_length)
    return stats["avg"] if average else stats["sum"]


@torch.no_grad()
def last_hidden(mdl, tok, text, max_length=None):
    """Mean-pooled last hidden state as a sentence embedding (numpy 1-D)."""
    max_length = max_length or MAX_LENGTH
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length).to(DEVICE)
    out = mdl(**enc, output_hidden_states=True)
    h = out.hidden_states[-1]
    mask = enc["attention_mask"].unsqueeze(-1).float()
    pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return pooled.squeeze(0).float().cpu().numpy()


@torch.no_grad()
def generate(mdl, tok, prompt, max_new_tokens=40):
    """Greedy generation. Returns the full decoded string (prompt + continuation)."""
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(DEVICE)
    out = mdl.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
    return tok.decode(out[0], skip_special_tokens=True)
