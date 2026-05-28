"""
Step 2 — Counterfactual Data Augmentation (CDA).
For every biography, create a gender-swapped counterfactual copy.
This version matches the final working notebook logic.
"""
import sys
import os
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Algorithm.config import DATA_DIR

import pandas as pd


def normalize_swap(text):
    t = str(text)
    swaps = {
        r"\bhe\b": "she",
        r"\bhim\b": "her",
        r"\bhis\b": "her",
        r"\bhimself\b": "herself",
        r"\bfather\b": "mother",
        r"\bhusband\b": "wife",
        r"\bson\b": "daughter",
        r"\bbrother\b": "sister",
        r"\bman\b": "woman",
        r"\bmen\b": "women",
        r"\bboy\b": "girl",
        r"\bboys\b": "girls",
        r"\bmr\.\b": "ms.",
        r"\bmr\b": "ms",
    }
    for pat, rep in swaps.items():
        t = re.sub(pat, rep, t, flags=re.I)
    return " ".join(t.split())


in_path = os.path.join(DATA_DIR, "bias_in_bios.csv")
out_path = os.path.join(DATA_DIR, "bias_in_bios_pairs.csv")

print("[Step 2] Building CDA pairs...")
df = pd.read_csv(in_path).dropna(subset=["text"]).copy()

df["text_cf"] = df["text"].apply(normalize_swap)
df["swap_changed"] = (df["text"] != df["text_cf"]).astype(int)

df.to_csv(out_path, index=False)

print(f"[Step 2] Saved pairs -> {out_path}")
print(f"[Step 2] Texts with at least one swap: {int(df['swap_changed'].sum())}/{len(df)}")
print(df[["text", "text_cf", "swap_changed"]].head(3).to_string())