"""Clickbait headline dataset (Chakraborty et al., 2016, "Stop Clickbait").

About 16k clickbait headlines (BuzzFeed, Upworthy, ViralNova, ...) and 16k
non clickbait headlines (WikiNews, New York Times, The Guardian, The Hindu).
Mirrored on GitHub by the authors' collaborators, so it downloads without an account.
"""
from __future__ import annotations

import gzip
import re
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

BASE = "https://raw.githubusercontent.com/bhargaviparanjape/clickbait/master/dataset"
FILES = {"clickbait_data.gz": 1, "non_clickbait_data.gz": 0}


def download(raw_dir: str | Path = "data/raw") -> Path:
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        if not (raw / name).exists():
            urllib.request.urlretrieve(f"{BASE}/{name}", raw / name)
    return raw


def normalise(text: str) -> str:
    """Lowercase and collapse whitespace.

    Lowercasing matters: the clickbait sources use Title Case while most news sources
    use sentence case, so a model fed raw casing can score well by learning the
    publisher's style guide instead of what makes a headline clickbait.
    """
    return re.sub(r"\s+", " ", text.strip().lower())


def load(raw_dir: str | Path = "data/raw", lowercase: bool = True) -> pd.DataFrame:
    raw = download(raw_dir)
    rows = []
    for name, label in FILES.items():
        with gzip.open(raw / name, "rt", encoding="utf-8", errors="replace") as f:
            rows += [(line.strip(), label) for line in f if line.strip()]
    df = pd.DataFrame(rows, columns=["raw_text", "label"])
    df["text"] = df["raw_text"].map(normalise) if lowercase else df["raw_text"]
    # Identical headlines can appear in both files; drop them so labels are unambiguous
    # and so the same text never lands in both train and test.
    df = df.drop_duplicates("text").reset_index(drop=True)
    return df


def split(df: pd.DataFrame, seed: int = 42):
    """Stratified 80 / 10 / 10 train, validation, test split."""
    train, rest = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["label"], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
