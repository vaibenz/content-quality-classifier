"""Score MIND news titles with the trained model, for use as a ranking feature.

  python scripts/score_mind.py --news ../content-recsys/data/MINDsmall/train/news.tsv \
      ../content-recsys/data/MINDsmall/dev/news.tsv --out ../content-recsys/data/mind_quality.tsv

Note the domain shift: the model was trained on pre 2017 headlines from specific
publishers, while MIND is 2019 MSN News. The script prints the score distribution so
you can sanity check it, and the README discusses what to look for.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quality.data import normalise  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--news", nargs="+", required=True)
    p.add_argument("--model", default="models/tfidf_logreg.pkl")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    with open(a.model, "rb") as f:
        bundle = pickle.load(f)
    frames = [pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 3], names=["news_id", "category", "title"],
                          quoting=3, dtype=str, keep_default_na=False) for path in a.news]
    news = pd.concat(frames).drop_duplicates("news_id")
    news["clickbait_prob"] = bundle["pipeline"].predict_proba(news["title"].map(normalise))[:, 1]
    news[["news_id", "clickbait_prob"]].to_csv(a.out, sep="\t", index=False, float_format="%.5f")

    print(f"scored {len(news):,} articles -> {a.out}")
    print(news["clickbait_prob"].describe().round(3).to_string())
    print("\nshare flagged at the precision threshold, by category:")
    flagged = news.assign(flag=news["clickbait_prob"] >= bundle["threshold"])
    print(flagged.groupby("category")["flag"].mean().sort_values(ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()
