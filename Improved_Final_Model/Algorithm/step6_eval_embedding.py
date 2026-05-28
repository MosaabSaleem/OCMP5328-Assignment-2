"""
Step 6 — Embedding-based bias evaluation.
Extracts the last hidden-state embeddings for each original/counterfactual
biography pair and computes cosine similarity between them.
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Algorithm.config import DATA_DIR, MODEL_DIR, METRICS_DIR, EVAL_SIZE
from Algorithm._model_helpers import load_model, last_hidden

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

pairs_path = os.path.join(DATA_DIR, "bias_in_bios_pairs.csv")
df_pairs = (
    pd.read_csv(pairs_path)
    .dropna(subset=["text", "text_cf"])
    .head(EVAL_SIZE)
)

for model_name in ["baseline", "debiased"]:
    print(f"\n[Step 6] Embedding eval — {model_name} ({len(df_pairs)} pairs)")
    mdl, tok = load_model(os.path.join(MODEL_DIR, model_name))

    rows = []
    for _, r in tqdm(df_pairs.iterrows(), total=len(df_pairs), desc="Embedding"):
        v1 = last_hidden(mdl, tok, r["text"])
        v2 = last_hidden(mdl, tok, r["text_cf"])
        cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
        rows.append(
            {
                "cosine_similarity": cos,
                "cosine_distance": 1.0 - cos,
            }
        )

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(METRICS_DIR, f"{model_name}_embedding.csv"), index=False)

    summary = {
        "model": model_name,
        "benchmark": "Embedding",
        "n": len(df_out),
        "mean_cosine_similarity": float(df_out["cosine_similarity"].mean()) if len(df_out) else None,
        "std_cosine_similarity": float(df_out["cosine_similarity"].std()) if len(df_out) else None,
        "mean_cosine_distance": float(df_out["cosine_distance"].mean()) if len(df_out) else None,
    }

    with open(os.path.join(METRICS_DIR, f"{model_name}_embedding_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    del mdl, tok
    print(
        f" mean cosine sim={summary['mean_cosine_similarity']:.6f} "
        f"dist={summary['mean_cosine_distance']:.6f}"
    )