"""
Step 2 — Counterfactual Data Augmentation (CDA).
For every biography, create a gender-swapped counterfactual copy by replacing
gendered pronouns and occupational nouns with their opposite-gender equivalent.
Both the original and counterfactual are kept in training (two-sided CDA).
Ref: Zhao et al., 2018. https://doi.org/10.18653/v1/N18-2003
     Zmigrod et al., 2019. https://doi.org/10.18653/v1/P19-1161
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Algorithm.config import DATA_DIR
from Algorithm._gender_swap import gender_swap

import pandas as pd

print("[Step 2] Building CDA pairs...")
df = pd.read_csv(os.path.join(DATA_DIR, "bias_in_bios.csv")).dropna(subset=["text"]).copy()
df["text_cf"]      = df["text"].astype(str).apply(gender_swap)
df["swap_changed"] = (df["text"].astype(str) != df["text_cf"].astype(str)).astype(int)

out_path = os.path.join(DATA_DIR, "bias_in_bios_pairs.csv")
df.to_csv(out_path, index=False)
changed = df["swap_changed"].sum()
print(f"[Step 2] Saved {len(df)} pairs -> {out_path}")
print(f"[Step 2] Texts with at least one swap: {changed}/{len(df)}")
print(df[["text","text_cf","swap_changed"]].head(3).to_string())
