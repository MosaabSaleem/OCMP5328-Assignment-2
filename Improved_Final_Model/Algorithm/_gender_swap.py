"""
Shared gender-swap helper.

Used by Step 2 (constructing CDA training pairs) and Step 4b (counterfactual
invariance evaluation). Keeping one implementation means evaluation always
measures the exact swap the model was trained on — if the swap rules drift
between training and eval, cf_gap becomes a different metric.

Ambiguous pronouns ("his", "her") are resolved by the next-token rule in
gender_swap(); everything else is a flat dictionary lookup. Swaps are applied
in one pass over the original tokens so chains like he->she->he never occur.
"""
import re

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
    """Match the capitalisation pattern of a replaced token."""
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement.capitalize()
    return replacement


def gender_swap(text: str) -> str:
    """
    Swap gendered terms in one pass over the original tokens so chains like
    he->she->he do not occur. Handles common possessive/object pronoun cases
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
