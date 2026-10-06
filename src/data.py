"""Dataset loading, cleaning and train/val/test splitting.

Pipeline (Section 6 of the project plan):
    1. Remove missing values
    2. Remove duplicate articles (prevents train/test leakage)
    3. Combine Title + Article body
    4. Convert labels  (Fake -> 0, Real -> 1)
    5. Stratified 80 / 10 / 10 split
"""
from __future__ import annotations

import re

import pandas as pd
from sklearn.model_selection import train_test_split
from .config import DATA_DIR, HF_DATASET, SEED


def set_seed(seed: int = SEED):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

CLEAN_CSV = DATA_DIR / "news_clean.csv"

# --- Text cleaning ---------------------------------------------------------
# The ISOT dataset has some well-known "shortcut" artifacts: almost every real
# article starts with "WASHINGTON (Reuters) -", and many fake ones end with
# "Featured image via ...". If we leave them in, the model learns the source
# signature instead of the language of the news, so we strip them.
_PATTERNS = [
    (re.compile(r"^.{0,120}?\(reuters\)\s*-\s*", re.I), ""),   # dateline prefix
    (re.compile(r"\breuters\b", re.I), ""),
    (re.compile(r"featured image via.*$", re.I | re.S), ""),
    (re.compile(r"(photo|image) (by|via|credit)[^.\n]*", re.I), ""),
    (re.compile(r"https?://\S+|www\.\S+|pic\.twitter\.com/\S+"), ""),
    (re.compile(r"@\w+"), ""),
    (re.compile(r"[\u2018\u2019]"), "'"),
    (re.compile(r"[\u201c\u201d]"), '"'),
    (re.compile(r"\ufffd"), "'"),
    (re.compile(r"\s+"), " "),
]


def clean_text(text: str) -> str:
    text = str(text)
    for pat, rep in _PATTERNS:
        text = pat.sub(rep, text)
    return text.strip()


def combine(title: str, body: str) -> str:
    """Title + Article body gives BERT more context."""
    title, body = clean_text(title), clean_text(body)
    if title and not title.endswith((".", "!", "?")):
        title += "."
    return f"{title} {body}".strip()


# --- Loading ---------------------------------------------------------------
def _load_raw() -> pd.DataFrame:
    """Prefer local Kaggle CSVs (data/Fake.csv + data/True.csv); otherwise
    download the public Hugging Face mirror."""
    fake_csv, true_csv = DATA_DIR / "Fake.csv", DATA_DIR / "True.csv"
    if fake_csv.exists() and true_csv.exists():
        print("[data] Using local Fake.csv / True.csv")
        fake = pd.read_csv(fake_csv).assign(label=0)
        real = pd.read_csv(true_csv).assign(label=1)
        return pd.concat([fake, real], ignore_index=True)

    print(f"[data] Downloading '{HF_DATASET}' from the Hugging Face Hub ...")
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET)
    return pd.concat([ds[s].to_pandas() for s in ds.keys()], ignore_index=True)


def build_clean_dataset(force: bool = False) -> pd.DataFrame:
    if CLEAN_CSV.exists() and not force:
        return pd.read_csv(CLEAN_CSV)

    df = _load_raw()
    n0 = len(df)

    # 1. Missing values
    df = df.dropna(subset=["text", "label"])
    df["title"] = df["title"].fillna("")
    df = df[df["text"].astype(str).str.strip().str.len() > 0]
    n1 = len(df)

    # 3. Combine title + body (done before dedup so we dedupe on final input)
    df["content"] = [combine(t, b) for t, b in zip(df["title"], df["text"])]
    df = df[df["content"].str.split().str.len() >= 5]

    # 2. Duplicates
    df = df.drop_duplicates(subset=["content"])
    n2 = len(df)

    # 4. Labels already 0 = Fake, 1 = Real
    df["label"] = df["label"].astype(int)
    keep = ["title", "content", "subject", "date", "label"]
    df = df[[c for c in keep if c in df.columns]].reset_index(drop=True)

    print(f"[data] raw={n0:,}  after-missing={n1:,}  after-dedup={n2:,}")
    df.to_csv(CLEAN_CSV, index=False)
    return df


def get_splits(sample: int | None = None, force: bool = False):
    """Return (train, val, test) DataFrames with an 80/10/10 stratified split.

    `sample` optionally limits the TOTAL number of rows (balanced, stratified)
    for fast CPU experiments.
    """
    df = build_clean_dataset(force=force)
    if sample and sample < len(df):
        df, _ = train_test_split(df, train_size=sample, stratify=df["label"], random_state=SEED)

    train, tmp = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
    val, test = train_test_split(tmp, test_size=0.5, stratify=tmp["label"], random_state=SEED)
    return (train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True))


if __name__ == "__main__":
    tr, va, te = get_splits(force=True)
    for name, d in (("train", tr), ("val", va), ("test", te)):
        print(f"{name:>5}: {len(d):>6,}  fake={int((d.label == 0).sum()):,}  real={int((d.label == 1).sum()):,}")
