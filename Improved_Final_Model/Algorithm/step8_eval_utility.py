"""
Step 8 — Utility evaluation.
Measures perplexity on WikiText-2 (language modelling quality does not degrade
after debiasing), generation speed, and records training time.
Required by assignment: utility metrics, average, standard deviation, training time.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Algorithm.config import MODEL_DIR, METRICS_DIR, EVAL_SIZE
from Algorithm._model_helpers import load_model, generate

import numpy as np
import torch
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
    """WikiText-2 perplexity. Lower = better language model quality."""
    ds    = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in ds["text"] if len(t.strip()) > 50][:n]
    nlls  = []
    for text in texts:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length)
        with torch.no_grad():
            nll = float(mdl(**enc, labels=enc["input_ids"]).loss)
        nlls.append(nll)
    return (round(float(np.exp(np.mean(nlls))), 4),
            round(float(np.mean(nlls)), 4),
            round(float(np.std(nlls)),  4))


for model_name in ["baseline", "debiased"]:
    print(f"\n[Step 8] Utility eval — {model_name}")
    mdl, tok = load_model(os.path.join(MODEL_DIR, model_name))

    ppl, mean_nll, std_nll = compute_perplexity(mdl, tok, n=EVAL_SIZE)

    gen_times = []
    for prompt in UTILITY_PROMPTS:
        t0 = time.time()
        generate(mdl, tok, prompt)
        gen_times.append(time.time() - t0)

    # Pull training time from saved JSON
    train_json = os.path.join(METRICS_DIR, f"train_{model_name}.json")
    train_secs = None
    if os.path.isfile(train_json):
        with open(train_json) as f:
            train_secs = json.load(f).get("train_seconds")

    summary = {
        "model": model_name,
        "benchmark": "Utility",
        "perplexity_wikitext2":        ppl,
        "mean_nll_loss":               mean_nll,
        "std_nll_loss":                std_nll,
        "mean_generation_seconds":     round(float(np.mean(gen_times)), 3),
        "std_generation_seconds":      round(float(np.std(gen_times)),  3),
        "train_seconds":               train_secs,
    }
    with open(os.path.join(METRICS_DIR, f"{model_name}_utility_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    del mdl, tok
    print(f"  PPL={ppl}  NLL mean={mean_nll} std={std_nll}  gen={summary['mean_generation_seconds']}s")