"""Train and compare the models, then write results/RESULTS.md, metrics.json, plots and models.

  python scripts/run_all.py            # full run, about 3 minutes on a laptop CPU
  python scripts/run_all.py --quick    # small subsample for CI
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quality import baseline, data, evaluate, textcnn  # noqa: E402

log = logging.getLogger("run_all")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results")
    p.add_argument("--models", default="models")
    p.add_argument("--target-precision", type=float, default=0.98)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    out, models_dir = Path(a.out), Path(a.models)
    out.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    df = data.load()
    if a.quick:
        df = df.groupby("label").sample(1500, random_state=a.seed)
    train, val, test = data.split(df, seed=a.seed)
    y_val, y_test = val["label"].to_numpy(), test["label"].to_numpy()
    res = {"data": {"n_total": len(df), "n_train": len(train), "n_val": len(val), "n_test": len(test),
                    "clickbait_share": float(df["label"].mean())}}

    # 1. TF-IDF + logistic regression
    lr, res["lr_tuning"] = baseline.fit_with_val(train, val)
    probs = {"tfidf_logreg": (lr.predict_proba(val["text"])[:, 1], lr.predict_proba(test["text"])[:, 1])}
    res["lr_top_features"] = dict(zip(("clickbait", "not_clickbait"), baseline.top_features(lr)))

    # 2. TextCNN from scratch, temperature scaled
    cnn, vocab, temp = textcnn.train(train, val, epochs=2 if a.quick else 8, seed=a.seed)
    probs["textcnn_uncalibrated"] = (textcnn.predict_proba(cnn, vocab, val["text"]),
                                     textcnn.predict_proba(cnn, vocab, test["text"]))
    probs["textcnn_calibrated"] = (textcnn.predict_proba(cnn, vocab, val["text"], temp),
                                   textcnn.predict_proba(cnn, vocab, test["text"], temp))
    res["textcnn_temperature"] = temp

    # 3. Shortcut check: what if we had kept the original casing?
    raw_tr = train.assign(text=train["raw_text"])
    raw_lr = baseline.make_pipeline(4.0, lowercase=False).fit(raw_tr["text"], raw_tr["label"])
    res["casing_shortcut"] = {
        "raw_case_model_on_raw_test_f1": evaluate.classification_report(
            y_test, raw_lr.predict_proba(test["raw_text"])[:, 1])["f1"],
        "raw_case_model_on_lowercased_test_f1": evaluate.classification_report(
            y_test, raw_lr.predict_proba(test["text"])[:, 1])["f1"],
    }

    res["models"] = {}
    for name, (pv, pt) in probs.items():
        thr = evaluate.threshold_for_precision(y_val, pv, a.target_precision)
        res["models"][name] = {"test@0.5": evaluate.classification_report(y_test, pt),
                               "test@precision_threshold": evaluate.classification_report(y_test, pt, thr)}

    # Error analysis on the deployed model (logistic regression, chosen for speed and transparency)
    pv, pt = probs["tfidf_logreg"]
    thr = res["models"]["tfidf_logreg"]["test@precision_threshold"]["threshold"]
    t = test.assign(prob=pt)
    fp = t[(t.label == 0) & (t.prob >= thr)].sort_values("prob", ascending=False).head(15)
    fn = t[(t.label == 1) & (t.prob < thr)].sort_values("prob").head(15)
    errors = ["# Error analysis (TF-IDF + LR, precision targeted threshold)", "",
              f"Threshold {thr:.3f}", "", "## News flagged as clickbait (false positives)", ""]
    errors += [f"- {p:.2f}  {s}" for s, p in zip(fp["raw_text"], fp["prob"])]
    errors += ["", "## Clickbait that slipped through (false negatives, lowest scores)", ""]
    errors += [f"- {p:.2f}  {s}" for s, p in zip(fn["raw_text"], fn["prob"])]
    (out / "errors.md").write_text("\n".join(errors) + "\n")

    plot(probs, y_test, out)
    with open(models_dir / "tfidf_logreg.pkl", "wb") as f:
        pickle.dump({"pipeline": lr, "threshold": thr}, f)
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    write_report(res, out)
    log.info("wrote %s", out)


def plot(probs, y_test, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, (_, pt) in probs.items():
        prec, rec, _ = precision_recall_curve(y_test, pt)
        axes[0].plot(rec, prec, label=name)
        conf, acc, _ = evaluate.reliability_curve(y_test, pt)
        axes[1].plot(conf, acc, marker="o", label=name)
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Precision vs recall (test)", ylim=(0.8, 1.01))
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set(xlabel="Predicted probability", ylabel="Observed clickbait rate", title="Reliability (test)")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=130)


def write_report(r: dict, out: Path):
    d = r["data"]
    lines = ["# Results", "",
             f"{d['n_total']:,} headlines after de-duplication ({d['clickbait_share']:.1%} clickbait), "
             f"split {d['n_train']:,} / {d['n_val']:,} / {d['n_test']:,} (train / val / test, stratified).", "",
             "## Test set, default threshold 0.5", "",
             "| Model | Accuracy | F1 | ROC AUC | PR AUC | Brier | ECE |", "|---|---|---|---|---|---|---|"]
    for name, m in r["models"].items():
        t = m["test@0.5"]
        lines.append(f"| {name} | {t['accuracy']:.4f} | {t['f1']:.4f} | {t['roc_auc']:.4f} | "
                     f"{t['pr_auc']:.4f} | {t['brier']:.4f} | {t['ece']:.4f} |")
    lines += ["", "## Test set, threshold chosen on validation for 98% precision", "",
              "| Model | Threshold | Precision | Recall | F1 |", "|---|---|---|---|---|"]
    for name, m in r["models"].items():
        t = m["test@precision_threshold"]
        lines.append(f"| {name} | {t['threshold']:.3f} | {t['precision']:.4f} | {t['recall']:.4f} | {t['f1']:.4f} |")
    c = r["casing_shortcut"]
    lines += ["", "## Casing shortcut check", "",
              f"A model trained on original casing scores F1 {c['raw_case_model_on_raw_test_f1']:.4f} on original "
              f"casing but {c['raw_case_model_on_lowercased_test_f1']:.4f} when the same test headlines are lowercased. "
              "The gap is how much it leaned on publisher style (Title Case) rather than wording, which is why the "
              "pipeline lowercases everything.", "",
              f"TextCNN temperature fitted on validation: {r['textcnn_temperature']:.3f}", "",
              "## Strongest n-grams (logistic regression)", "",
              "Clickbait: " + ", ".join(f"`{f}`" for f, _ in r["lr_top_features"]["clickbait"][:12]), "",
              "Not clickbait: " + ", ".join(f"`{f}`" for f, _ in r["lr_top_features"]["not_clickbait"][:12]), "",
              "![curves](curves.png)", "", "See [errors.md](errors.md) for misclassified headlines."]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
