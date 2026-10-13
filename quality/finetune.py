"""Optional: fine-tune DistilBERT on the same splits.

Needs `pip install transformers` and internet access to the Hugging Face Hub.
Runs on CPU in roughly 30 to 60 minutes for 2 epochs; on a free Colab GPU in a few minutes.

  python -m quality.finetune --out results/distilbert
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import data
from .evaluate import classification_report, threshold_for_precision

log = logging.getLogger(__name__)


def encode(tok, texts, max_len=48):
    return tok(list(texts), truncation=True, max_length=max_len, padding="max_length",
               return_tensors="pt")


@torch.no_grad()
def probs(model, enc, device, batch=256):
    model.eval()
    out = []
    for s in range(0, len(enc["input_ids"]), batch):
        b = {k: v[s:s + batch].to(device) for k, v in enc.items()}
        out.append(torch.softmax(model(**b).logits, -1)[:, 1].cpu())
    return torch.cat(out).numpy()


def main():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="distilbert-base-uncased")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", default="results/distilbert")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train, val, test = data.split(data.load(), seed=a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=2).to(device)
    enc_tr, enc_va, enc_te = (encode(tok, d["text"]) for d in (train, val, test))
    y_tr = torch.tensor(train["label"].to_numpy())

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * ((len(train) + a.batch_size - 1) // a.batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / steps))
    for epoch in range(a.epochs):
        model.train()
        perm = torch.randperm(len(train))
        for i, s in enumerate(range(0, len(perm), a.batch_size)):
            b = perm[s:s + a.batch_size]
            batch = {k: v[b].to(device) for k, v in enc_tr.items()}
            loss = F.cross_entropy(model(**batch).logits, y_tr[b].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if i % 100 == 0:
                log.info("epoch %d step %d loss %.4f", epoch + 1, i, loss.item())

    p_val, p_test = probs(model, enc_va, device), probs(model, enc_te, device)
    thr = threshold_for_precision(val["label"].to_numpy(), p_val)
    report = {"test@0.5": classification_report(test["label"].to_numpy(), p_test),
              "test@precision_threshold": classification_report(test["label"].to_numpy(), p_test, thr)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    np.save(out / "test_probs.npy", p_test)
    log.info(json.dumps(report["test@0.5"], indent=2))


if __name__ == "__main__":
    main()
