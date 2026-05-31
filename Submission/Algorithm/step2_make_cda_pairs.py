"""
Step 2 — Counterfactual Data Augmentation (CDA)
For every biography, create a gender-swapped counterfactual copy by replacing
gendered pronouns and occupational nouns with their opposite gender equivalent.
Both the original and counterfactual are kept in training (two-sided CDA).
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Algorithm.config import DATA_DIR
from Algorithm._gender_swap import gender_swap

import pandas as pd

#Helper to normalize and knowingly swap common gendered terms. This is not exhaustive but should cover most cases in the dataset.
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

#Load the original data, create counterfactuals, and save the new pairs dataset
in_path = os.path.join(DATA_DIR, "bias_in_bios.csv")
out_path = os.path.join(DATA_DIR, "bias_in_bios_pairs.csv")

print("[Step 2] Building CDA pairs...")
df = pd.read_csv(in_path).dropna(subset=["text"]).copy()

#Apply the normalize_swap function to create counterfactual texts, and track which rows had any changes
df["text_cf"] = df["text"].apply(normalize_swap)
df["swap_changed"] = (df["text"] != df["text_cf"]).astype(int)

#Keep only rows where a swap was made, and save the new dataset with original and counterfactual pairs
df.to_csv(out_path, index=False)
changed = df["swap_changed"].sum()
print(f"[Step 2] Saved {len(df)} pairs -> {out_path}")
print(f"[Step 2] Texts with at least one swap: {changed}/{len(df)}")
print(df[["text","text_cf","swap_changed"]].head(3).to_string())
