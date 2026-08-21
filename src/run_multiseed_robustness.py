
"""Retrain CNN and BERT across three seeds to quantify run-to-run variance."""
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import repro
import config as cfg
from metrics import compute_metrics
from preprocessing import clean_series
from train import CNNDataset, TextCNN, BertDataset, get_cnn_vocab_and_embed

EXTRA_DIR = cfg.RESULTS_DIR / "extra"
EXTRA_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COMPS = ["real_real", "half_synthetic", "full_synthetic", "synthetic_25pct", "synthetic_75pct",
         "synthetic_real_only", "both_synthetic",
         "style_robust", "synthetic_multisource"]
SEEDS = [42, 1, 2]


def load_tests():
    out = {}
    for agg, key in ((True, "aggressive"), (False, "minimal")):
        d = {}
        for t in ("test_indomain", "test_crossdomain"):
            df = pd.read_csv(cfg.PROCESSED_DIR / f"{t}.csv")
            df["clean"] = clean_series(df["text"], aggressive=agg)
            d[t.replace("test_", "")] = df
        out[key] = d
    return out


def run_cnn(tests, rows):
    vocab, embed_base = get_cnn_vocab_and_embed()

    for comp in COMPS:
        for seed in SEEDS:
            repro.set_determinism(seed)
            embed = embed_base.clone()
            tr = pd.read_csv(cfg.PROCESSED_DIR / f"train_{comp}.csv")
            tr["clean"] = clean_series(tr["text"])
            gen = torch.Generator().manual_seed(seed)
            dl = DataLoader(CNNDataset(tr["clean"], tr["label"], vocab, cfg.CNN_MAX_LEN),
                             batch_size=cfg.CNN_BATCH_SIZE, shuffle=True, generator=gen)
            model = TextCNN(embed, cfg.CNN_NUM_FILTERS, cfg.CNN_FILTER_SIZES, cfg.CNN_DROPOUT).to(DEVICE)
            opt = torch.optim.Adam(model.parameters(), lr=cfg.CNN_LR)
            crit = nn.CrossEntropyLoss()
            model.train()
            for _ in range(cfg.CNN_EPOCHS):
                for xb, yb in dl:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()

            model.eval()
            for test_label, test_df in tests["aggressive"].items():
                tdl = DataLoader(CNNDataset(test_df["clean"], test_df["label"], vocab, cfg.CNN_MAX_LEN),
                                  batch_size=cfg.CNN_BATCH_SIZE)
                probs, preds, ys = [], [], []
                with torch.no_grad():
                    for xb, yb in tdl:
                        logits = model(xb.to(DEVICE))
                        p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                        probs.extend(p); preds.extend(logits.argmax(1).cpu().numpy()); ys.extend(yb.numpy())
                m = compute_metrics(np.array(ys), np.array(preds), np.array(probs))
                rows.append({"model": "CNN", "comp": comp, "seed": seed, "test": test_label,
                             "f1": round(m["f1"], 4), "auc_roc": round(m["auc_roc"], 4)})
            print(f"  [CNN] {comp} seed={seed} done")


def run_bert(tests, rows, grad_accum=1):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    tok = AutoTokenizer.from_pretrained(cfg.BERT_MODEL_NAME)

    for comp in COMPS:
        for seed in SEEDS:
            repro.set_determinism(seed)
            tr = pd.read_csv(cfg.PROCESSED_DIR / f"train_{comp}.csv")
            tr["clean"] = clean_series(tr["text"], aggressive=False)
            gen = torch.Generator().manual_seed(seed)
            dl = DataLoader(BertDataset(tr["clean"], tr["label"], tok, cfg.BERT_MAX_LEN),
                             batch_size=cfg.BERT_BATCH_SIZE, shuffle=True, generator=gen)
            model = AutoModelForSequenceClassification.from_pretrained(
                cfg.BERT_MODEL_NAME, num_labels=2).to(DEVICE)
            opt = torch.optim.AdamW(model.parameters(), lr=cfg.BERT_LR)
            scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))
            steps_per_epoch = -(-len(dl) // grad_accum)
            total_steps = steps_per_epoch * cfg.BERT_EPOCHS
            warmup_steps = max(1, int(0.1 * total_steps))
            sched = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)
            model.train()
            for _ in range(cfg.BERT_EPOCHS):
                opt.zero_grad()
                for step, batch in enumerate(dl):
                    batch = {k: v.to(DEVICE) for k, v in batch.items()}
                    with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                        loss = model(**batch).loss / grad_accum
                    scaler.scale(loss).backward()
                    if (step + 1) % grad_accum == 0:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()

            model.eval()
            for test_label, test_df in tests["minimal"].items():
                tdl = DataLoader(BertDataset(test_df["clean"], test_df["label"], tok, cfg.BERT_MAX_LEN),
                                  batch_size=cfg.BERT_BATCH_SIZE)
                probs, preds, ys = [], [], []
                with torch.no_grad():
                    for batch in tdl:
                        y = batch.pop("labels")
                        batch = {k: v.to(DEVICE) for k, v in batch.items()}
                        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                            logits = model(**batch).logits
                        probs.extend(torch.softmax(logits.float(), 1)[:, 1].cpu().numpy())
                        preds.extend(logits.argmax(1).cpu().numpy()); ys.extend(y.numpy())
                m = compute_metrics(np.array(ys), np.array(preds), np.array(probs))
                rows.append({"model": "BERT", "comp": comp, "seed": seed, "test": test_label,
                             "f1": round(m["f1"], 4), "auc_roc": round(m["auc_roc"], 4)})
            print(f"  [BERT] {comp} seed={seed} done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", choices=["cnn", "bert"], default=["cnn", "bert"])
    ap.add_argument("--comps", nargs="+", default=None,
                    help="subset of COMPS to run (default: all). Results merge into the "
                         "existing CSV, so re-running everything just to add one "
                         "composition wastes hours of GPU time for no new information.")
    args = ap.parse_args()

    if args.comps:
        unknown = set(args.comps) - set(COMPS)
        if unknown:
            ap.error(f"unknown composition(s): {sorted(unknown)}. Known: {COMPS}")
        COMPS[:] = args.comps

    print(f"Device: {DEVICE}")
    print(f"Compositions this run: {COMPS}")
    tests = load_tests()
    rows = []
    if "cnn" in args.models:
        run_cnn(tests, rows)
    if "bert" in args.models:
        run_bert(tests, rows)

    df = pd.DataFrame(rows)
    out = EXTRA_DIR / "multiseed_results.csv"
    if out.exists():
        old = pd.read_csv(out)
        df = pd.concat([old, df], ignore_index=True).drop_duplicates(
            subset=["model", "comp", "seed", "test"], keep="last")
    df.to_csv(out, index=False)

    print("\n=== MEAN +/- STD (crossdomain F1, across seeds 42/1/2) ===")
    cross = df[df.test == "crossdomain"]
    summary = cross.groupby(["model", "comp"])["f1"].agg(["mean", "std"]).round(4)
    print(summary.to_string())
    print(f"\nSaved per-seed rows to {out}")


if __name__ == "__main__":
    main()
