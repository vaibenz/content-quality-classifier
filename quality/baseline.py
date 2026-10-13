"""Strong classical baseline: TF-IDF word and character n-grams + logistic regression."""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import FeatureUnion, Pipeline


def make_pipeline(C: float = 4.0, lowercase: bool = True) -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=lowercase)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True,
                                 lowercase=lowercase)),
    ])
    return Pipeline([("tfidf", features), ("lr", LogisticRegression(C=C, max_iter=2000))])


def fit_with_val(train, val, Cs=(1.0, 4.0, 16.0, 64.0, 256.0)):
    """Pick regularisation strength on validation log loss, then return the fitted pipeline."""
    best, best_loss, history = None, np.inf, []
    for C in Cs:
        pipe = make_pipeline(C).fit(train["text"], train["label"])
        loss = log_loss(val["label"], pipe.predict_proba(val["text"])[:, 1])
        history.append({"C": C, "val_log_loss": float(loss)})
        if loss < best_loss:
            best, best_loss = pipe, loss
    return best, history


def top_features(pipe: Pipeline, k: int = 15):
    names = pipe.named_steps["tfidf"].get_feature_names_out()
    coef = pipe.named_steps["lr"].coef_[0]
    order = np.argsort(coef)
    clean = lambda i: names[i].split("__", 1)[1]  # noqa: E731
    return ([(clean(i), float(coef[i])) for i in order[::-1][:k]],
            [(clean(i), float(coef[i])) for i in order[:k]])
