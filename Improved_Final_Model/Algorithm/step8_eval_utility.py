"""
Step 8 — Utility evaluation.
Measures perplexity on WikiText-2, generation speed, and training time.
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Algorithm.config import MODEL_DIR, METRICS_DIR, EVAL_SIZE
from Algorithm._model_helpers import load_model, generate

import numpy as np
import torch
from tqdm.auto import tqdm
from datasets import load_dataset

UTILITY_PROMPTS = [
    "The doctor examined the patient and",
    "The nurse carefully prepared the",
    "The engineer designed the new system to",
    "The teacher explained the lesson about",
    "The CEO announced that the company would",
    "The software developer wrote code that",
]


def compute_perplexity(mdl, tok, n=50, max_length=128):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in ds["text"] if len(t.strip()) > 50][:n]

    nlls = []
    for text in tqdm(texts, desc="Perplexity"):
        enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length)
        enc = {k: v.to(mdl.device) for k, v in enc.items()}
        with torch.no_grad():
            nll = float(mdl(**enc, labels=enc["input_ids"]).loss)
        nlls.append(nll)

    return {
        "perplexity_wikitext2": float(np.exp(np.mean(nlls))),
        "mean_nll_loss": float(np.mean(nlls)),
        "std_nll_loss": float(np.std(nlls)),
    }


def compute_generation_speed(mdl, tok):
    gen_times = []
    for prompt in UTILITY_PROMPTS:
        t0 = time.time()
        generate(mdl, tok, prompt)
        gen_times.append(time.time() - t0)

    return {
        "mean_generation_seconds": float(np.mean(gen_times)),
        "std_generation_seconds": float(np.std(gen_times)),
    }


for model_name in ["baseline", "debiased"]:
    print(f"\n[Step 8] Utility eval — {model_name}")
    mdl, tok = load_model(os.path.join(MODEL_DIR, model_name))

    util = compute_perplexity(mdl, tok, n=max(5, EVAL_SIZE))
    util.update(compute_generation_speed(mdl, tok))

    train_json = os.path.join(METRICS_DIR, f"train_{model_name}.json")
    if os.path.isfile(train_json):
        with open(train_json) as f:
            util["train_seconds"] = json.load(f).get("train_seconds")
    else:
        util["train_seconds"] = None

    summary = {
        "model": model_name,
        "benchmark": "Utility",
        **util,
    }

    with open(os.path.join(METRICS_DIR, f"{model_name}_utility_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    del mdl, tok
    print(
        f" PPL={summary['perplexity_wikitext2']:.4f} "
        f"NLL mean={summary['mean_nll_loss']:.4f} "
        f"std={summary['std_nll_loss']:.4f} "
        f"gen={summary['mean_generation_seconds']:.3f}s"
    )