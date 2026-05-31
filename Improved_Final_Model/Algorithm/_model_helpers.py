"""
_model_helpers.py
Shared utilities used by all evaluation steps:
- load_model : loads base or LoRA-adapted model
- seq_logprob : full-sequence log-probability
- generate : greedy text generation
- last_hidden : mean-pooled last hidden state (sentence embedding)
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer, AutoModelForCausalLM

from huggingface_hub import login
#Please add your own token here. My token gets revoked too often so it may not be active when you run this code. You can create your own token at https://huggingface.co/settings/tokens
login("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")

#load model from path, handling both base and LoRA-adapted models
def load_model(path: str):
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if os.path.isfile(os.path.join(path, "adapter_config.json")):
        from peft import PeftModel
        from Algorithm.config import MODEL_NAME

        base = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        mdl = PeftModel.from_pretrained(base, path)
    else:
        mdl = AutoModelForCausalLM.from_pretrained(path)

    mdl.eval()
    return mdl, tok

#Return the sum log-probability of a full sequence 
def seq_logprob(mdl, tok, text: str, max_length: int = 256) -> float:
    enc = tok(str(text), return_tensors="pt", truncation=True, max_length=max_length)
    enc = {k: v.to(mdl.device) for k, v in enc.items()}
    with torch.no_grad():
        loss = mdl(**enc, labels=enc["input_ids"]).loss
    return -float(loss) * enc["input_ids"].shape[1]

#Greedy decode a continuation from prompt
def generate(mdl, tok, prompt: str, max_new_tokens: int = 40) -> str:
    enc = tok(str(prompt), return_tensors="pt", truncation=True, max_length=256)
    enc = {k: v.to(mdl.device) for k, v in enc.items()}
    with torch.no_grad():
        out = mdl.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0], skip_special_tokens=True)

#Mean-pool the last hidden state as a sentence embedding
def last_hidden(mdl, tok, text: str, max_length: int = 128):
    enc = tok(str(text), return_tensors="pt", truncation=True, max_length=max_length)
    enc = {k: v.to(mdl.device) for k, v in enc.items()}
    with torch.no_grad():
        if str(mdl.device).startswith("cuda"):
            with torch.autocast(device_type="cuda", enabled=False):
                out = mdl(**enc, output_hidden_states=True)
        else:
            out = mdl(**enc, output_hidden_states=True)
        hs = out.hidden_states[-1].float()
    return hs.mean(dim=1).squeeze(0).detach().cpu().numpy()