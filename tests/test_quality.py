import numpy as np
import pandas as pd
import pytest

from quality import baseline, data, evaluate, textcnn

CLICKBAIT = ["you won't believe what happened next", "17 things only cat owners understand",
             "this one trick will change your life", "here's why everyone is talking about this",
             "21 photos that will make you smile", "what happens next will shock you"]
NEWS = ["senate passes budget bill after long debate", "central bank holds interest rates steady",
        "storm causes flooding across northern region", "company reports quarterly earnings rise",
        "court rules on disputed land case", "minister announces new transport plan"]


def toy(n=20):
    rows = [(t, 1) for t in CLICKBAIT] * n + [(t, 0) for t in NEWS] * n
    return pd.DataFrame(rows, columns=["text", "label"]).sample(frac=1, random_state=0)


def test_normalise_lowercases_and_collapses_space():
    assert data.normalise("  You WON'T   Believe ") == "you won't believe"


def test_threshold_meets_target_precision_on_validation():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = np.clip(y * 0.6 + rng.random(2000) * 0.5, 0, 1)
    thr = evaluate.threshold_for_precision(y, p, 0.95)
    pred = p >= thr
    assert (y[pred] == 1).mean() >= 0.95


def test_ece_is_zero_for_perfect_calibration():
    p = np.repeat([0.2, 0.8], 5000)
    y = np.concatenate([np.r_[np.ones(1000), np.zeros(4000)], np.r_[np.ones(4000), np.zeros(1000)]])
    assert evaluate.expected_calibration_error(y, p) == pytest.approx(0.0, abs=1e-9)


def test_baseline_learns_toy_task():
    df = toy()
    pipe = baseline.make_pipeline().fit(df["text"], df["label"])
    assert pipe.predict_proba(["you won't believe these 17 photos"])[0, 1] > 0.5
    assert pipe.predict_proba(["senate passes transport bill"])[0, 1] < 0.5


def test_textcnn_trains_and_calibrates():
    df = toy()
    model, vocab, temp = textcnn.train(df, df, epochs=3, seed=0)
    p = textcnn.predict_proba(model, vocab, df["text"], temp)
    assert temp > 0
    assert ((p > 0.5) == df["label"].to_numpy()).mean() > 0.9
