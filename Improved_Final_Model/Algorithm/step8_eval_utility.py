"""
Step 8 — Utility evaluation.
Measures perplexity on WikiText-2 (language modelling quality does not degrade
after debiasing), generation speed, and records training time.
Required by assignment: utility metrics, average, standard deviation, training time.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Algorithm.config import METRICS_DIR, EVAL_SAMPLE_SIZE, MODELS_TO_EVAL
from Algorithm._model_helpers import load_model, generate, resolve_model_path
from Algorithm._stats import bootstrap_ci
from Algorithm._wandb_log import start as wandb_start, finish as wandb_finish

import numpy as np
import pandas as pd
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
    ds     = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    texts  = [t for t in ds["text"] if len(t.strip()) > 50][:n]
    device = next(mdl.parameters()).device
    nlls   = []
    for text in texts:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            nll = float(mdl(**enc, labels=enc["input_ids"]).loss)
        nlls.append(nll)
    nll_ci = bootstrap_ci(nlls)
    ppl_ci = (
        round(float(np.exp(nll_ci["ci_low"])),  4) if nll_ci["ci_low"]  is not None else None,
        round(float(np.exp(nll_ci["ci_high"])), 4) if nll_ci["ci_high"] is not None else None,
    )
    return (round(float(np.exp(np.mean(nlls))), 4),
            round(float(np.mean(nlls)), 4),
            round(float(np.std(nlls)),  4),
            nll_ci,
            ppl_ci)


wb_run = wandb_start(
    job_type="eval",
    name="utility",
    config={
        "benchmark":         "WikiText-2 perplexity + generation speed",
        "n_wikitext_chunks": EVAL_SAMPLE_SIZE,
        "n_gen_prompts":     len(UTILITY_PROMPTS),
    },
)
util_rows = []

for model_name in MODELS_TO_EVAL:
    print(f"\n[Step 8] Utility eval — {model_name}")
    mdl, tok = load_model(resolve_model_path(model_name))

    ppl, mean_nll, std_nll, nll_ci, ppl_ci = compute_perplexity(mdl, tok, n=EVAL_SAMPLE_SIZE)

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
        "perplexity_wikitext2_ci":     [ppl_ci[0], ppl_ci[1]],
        "mean_nll_loss":               mean_nll,
        "mean_nll_loss_ci":            [nll_ci["ci_low"], nll_ci["ci_high"]],
        "std_nll_loss":                std_nll,
        "mean_generation_seconds":     round(float(np.mean(gen_times)), 3),
        "std_generation_seconds":      round(float(np.std(gen_times)),  3),
        "train_seconds":               train_secs,
    }
    with open(os.path.join(METRICS_DIR, f"{model_name}_utility_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if wb_run is not None:
        wb_run.summary[f"perplexity/{model_name}"]   = ppl
        wb_run.summary[f"nll_mean/{model_name}"]     = mean_nll
        wb_run.summary[f"nll_std/{model_name}"]      = std_nll
        wb_run.summary[f"gen_seconds/{model_name}"]  = summary["mean_generation_seconds"]
        wb_run.summary[f"train_seconds/{model_name}"] = train_secs
        util_rows.append({
            "model":             model_name,
            "perplexity":        ppl,
            "nll_mean":          mean_nll,
            "nll_std":           std_nll,
            "gen_seconds_mean":  summary["mean_generation_seconds"],
            "gen_seconds_std":   summary["std_generation_seconds"],
            "train_seconds":     train_secs,
        })

    del mdl, tok
    print(f"  PPL={ppl}  NLL mean={mean_nll} std={std_nll}  gen={summary['mean_generation_seconds']}s")

if wb_run is not None and util_rows:
    import wandb
    wb_run.log({"utility_summary": wandb.Table(dataframe=pd.DataFrame(util_rows))})
wandb_finish(wb_run)
