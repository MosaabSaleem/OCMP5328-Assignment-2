"""
Direct downloaders for eval datasets that fail under datasets>=4 (script-based)
or have metadata incompatible with our pinned datasets version. Files are
cached under Input_data/cache/ so they download only once.
Each loader tries a list of candidate URLs and uses the first that works.
"""
import os
import csv
import json
import re
import urllib.request

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Algorithm.config import DATA_DIR

CACHE_DIR = os.path.join(DATA_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _try_download(urls, dest):
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        return dest
    last_err = None
    for url in urls:
        try:
            print(f"  Downloading {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as out:
                out.write(resp.read())
            if os.path.getsize(dest) > 0:
                return dest
        except Exception as e:
            print(f"    failed: {e}")
            last_err = e
            if os.path.isfile(dest):
                os.remove(dest)
    raise RuntimeError(f"All download URLs failed. Last error: {last_err}")


def load_crowspairs(n, bias_type=None):
    """
    CrowS-Pairs test set. Returns a list of dicts with keys:
      sent_more, sent_less, bias_type, stereo_antistereo
    If bias_type is given (e.g. 'gender'), filters to that subset first.
    """
    urls = [
        "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv",
    ]
    path = _try_download(urls, os.path.join(CACHE_DIR, "crows_pairs_anonymized.csv"))
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if bias_type and r.get("bias_type", "").strip().lower() != bias_type.lower():
                continue
            rows.append({
                "sent_more":         r.get("sent_more", ""),
                "sent_less":         r.get("sent_less", ""),
                "bias_type":         r.get("bias_type", ""),
                "stereo_antistereo": r.get("stereo_antistereo", ""),
            })
    return rows[:n] if n and n < len(rows) else rows


def load_stereoset_intrasentence(n, bias_type=None):
    """
    StereoSet intrasentence dev split. Returns a list of dicts with keys:
      context, sentences (each with gold_label, sentence), bias_type
    If bias_type is given (e.g. 'gender'), filters to that subset first.
    """
    urls = [
        "https://raw.githubusercontent.com/moinnadeem/StereoSet/master/data/dev.json",
        "https://raw.githubusercontent.com/McGill-NLP/bias-bench/main/data/stereoset/dev.json",
        "https://huggingface.co/datasets/McGill-NLP/stereoset/resolve/main/data/dev.json",
    ]
    path = _try_download(urls, os.path.join(CACHE_DIR, "stereoset_dev.json"))
    with open(path) as f:
        data = json.load(f)
    items = data["data"]["intrasentence"]
    if bias_type:
        items = [x for x in items if x.get("bias_type", "").strip().lower() == bias_type.lower()]
    return items[:n] if n and n < len(items) else items


def load_winobias_type1(n_per_side):
    """
    WinoBias type-1 test set, both pro- and anti-stereotype.
    Returns a list of {sentence, type} dicts where type is
    'type1_pro' or 'type1_anti'. Strips [coreference] brackets.
    """
    base_candidates = [
        "https://raw.githubusercontent.com/uclanlp/corefBias/master/WinoBias/wino/data",
        "https://raw.githubusercontent.com/uclanlp/corefBias/main/WinoBias/wino/data",
    ]
    files = {
        "type1_pro":  ("pro_stereotyped_type1.txt.test",  "winobias_pro_type1.txt"),
        "type1_anti": ("anti_stereotyped_type1.txt.test", "winobias_anti_type1.txt"),
    }
    rows = []
    for typ, (remote, local) in files.items():
        urls = [f"{b}/{remote}" for b in base_candidates]
        path = _try_download(urls, os.path.join(CACHE_DIR, local))
        with open(path) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if n_per_side and n_per_side < len(lines):
            lines = lines[:n_per_side]
        for ln in lines:
            ln = re.sub(r"^\d+\s+", "", ln)
            ln = ln.replace("[", "").replace("]", "").strip()
            if ln:
                rows.append({"sentence": ln, "type": typ})
    return rows


def load_bold_gender(n):
    """
    BOLD gender-prompt subset. Returns a list of {prompt, domain} dicts.
    """
    urls = [
        "https://raw.githubusercontent.com/amazon-science/bold/main/prompts/gender_prompt.json",
        "https://raw.githubusercontent.com/amazon-science/bold/master/prompts/gender_prompt.json",
    ]
    path = _try_download(urls, os.path.join(CACHE_DIR, "bold_gender_prompt.json"))
    with open(path) as f:
        data = json.load(f)
    rows = []
    for domain, people in data.items():
        for _person, prompts in people.items():
            for p in prompts:
                rows.append({"prompt": p, "domain": domain})
                if n and len(rows) >= n:
                    return rows
    return rows
