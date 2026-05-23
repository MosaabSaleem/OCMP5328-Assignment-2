"""
Step 6 — Embedding-based bias evaluation.
Extracts the last hidden-state embeddings for each original/counterfactual
biography pair and computes cosine similarity between them.
A debiased model should produce MORE similar representations for gender-swapped
sentences (higher cosine similarity = less gender separation in embedding space).
Covers: Assignment 'embedding-based metrics' category.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Algorithm.config import DATA_DIR, MODEL_DIR, METRICS_DIR, EVAL_SIZE
from Algorithm._model_helpers import load_model, last_hidden

import numpy as np
import pandas as pd

# Use the CDA pairs so we always compare original vs counterfactual
df_pairs = (pd.read_csv(os.path.join(DATA_DIR, "bias_in_bios_pairs.csv"))
              .dropna(subset=["text", "text_cf"])
              .head(EVAL_SIZE))

for model_name in ["baseline", "debiased"]:
    print(f"\n[Step 6] Embedding eval — {model_name}  ({len(df_pairs)} pairs)")
    mdl, tok = load_model(os.path.join(MODEL_DIR, model_name))

    rows = []
    for _, r in df_pairs.iterrows():
        v1  = last_hidden(mdl, tok, r["text"])
        v2  = last_hidden(mdl, tok, r["text_cf"])
        cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
        rows.append({
            "cosine_similarity": round(cos, 6),
            "cosine_distance":   round(1.0 - cos, 6),
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(METRICS_DIR, f"{model_name}_embedding.csv"), index=False)
    summary = {
        "model": model_name,
        "benchmark": "Embedding cosine (Bias-in-Bios pairs)",
        "n": len(df_out),
        "mean_cosine_similarity": round(float(df_out["cosine_similarity"].mean()), 4),
        "std_cosine_similarity":  round(float(df_out["cosine_similarity"].std()),  4),
        "mean_cosine_distance":   round(float(df_out["cosine_distance"].mean()),   4),
    }
    with open(os.path.join(METRICS_DIR, f"{model_name}_embedding_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    del mdl, tok
    print(f"  mean cosine sim={summary['mean_cosine_similarity']}  dist={summary['mean_cosine_distance']}")