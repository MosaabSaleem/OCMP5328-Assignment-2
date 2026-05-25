"""
Step 2 — Counterfactual Data Augmentation (CDA).
For every biography, create a gender-swapped counterfactual copy by replacing
gendered pronouns and occupational nouns with their opposite-gender equivalent.
Both the original and counterfactual are kept in training (two-sided CDA).
Ref: Zhao et al., 2018. https://doi.org/10.18653/v1/N18-2003
     Zmigrod et al., 2019. https://doi.org/10.18653/v1/P19-1161
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Algorithm.config import DATA_DIR

import pandas as pd

# Gender swap terms. Ambiguous pronouns such as "her" and "his" are handled
# separately in gender_swap() because their correct replacement depends on use.
SIMPLE_SWAPS = {
    "he": "she",
    "she": "he",
    "him": "her",
    "hers": "his",
    "himself": "herself",
    "herself": "himself",
    "man": "woman",
    "woman": "man",
    "men": "women",
    "women": "men",
    "male": "female",
    "female": "male",
    "boy": "girl",
    "girl": "boy",
    "boys": "girls",
    "girls": "boys",
    "father": "mother",
    "mother": "father",
    "husband": "wife",
    "wife": "husband",
    "son": "daughter",
    "daughter": "son",
    "brother": "sister",
    "sister": "brother",
    "businessman": "businesswoman",
    "businesswoman": "businessman",
    "actor": "actress",
    "actress": "actor",
    "waiter": "waitress",
    "waitress": "waiter",
    "mr": "ms",
    "ms": "mr",
    "mrs": "mr",
}

OBJECT_HER_FOLLOWERS = {
    "a", "an", "the", "this", "that", "these", "those",
    "to", "and", "or", "but", "because", "while", "when",
    "after", "before", "as", "if", "than",
    "in", "on", "at", "by", "for", "from", "of", "with",
    "without", "into", "onto", "over", "under", "through",
    "again", "today", "yesterday", "tomorrow",
}


def preserve_case(source: str, replacement: str) -> str:
    """Match the capitalization pattern of a replaced token."""
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement.capitalize()
    return replacement

def gender_swap(text: str) -> str:
    """
    Swap gendered terms in one pass over the original tokens so chains like
    he->she->he do not occur. Handles common possessive/object pronoun cases:
      his book -> her book, the book is his -> the book is hers
      her book -> his book, spoke to her -> spoke to him
    """
    original = str(text)
    tokens = list(re.finditer(r"\b[A-Za-z]+\b", original))
    if not tokens:
        return original

    out = []
    cursor = 0
    for i, match in enumerate(tokens):
        word = match.group(0)
        lower = word.lower()
        next_match = tokens[i + 1] if i + 1 < len(tokens) else None
        next_word = next_match.group(0).lower() if next_match else None
        next_is_adjacent = (
            next_match is not None
            and original[match.end():next_match.start()].strip() == ""
        )

        replacement = SIMPLE_SWAPS.get(lower)
        if lower == "his":
            replacement = "her" if next_is_adjacent and next_word not in OBJECT_HER_FOLLOWERS else "hers"
        elif lower == "her":
            replacement = "his" if next_is_adjacent and next_word not in OBJECT_HER_FOLLOWERS else "him"

        out.append(original[cursor:match.start()])
        out.append(preserve_case(word, replacement) if replacement else word)
        cursor = match.end()

    out.append(original[cursor:])
    return "".join(out)

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
