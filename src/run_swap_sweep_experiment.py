
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader

import repro
import config as cfg
from metrics import compute_metrics
from preprocessing import clean_series

EXTRA_DIR = cfg.RESULTS_DIR / "extra"
EXTRA_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# (sweep label, actual filename stem) -- 0/50/100 reuse already-built files
SWEEP_POINTS = [
    ("swap_000", "real_real"),
    ("swap_025", "swap_025"),
    ("swap_050", "mixed"),
    ("swap_075", "swap_075"),
    ("swap_100", "real_syn"),
]


def load_tests(aggressive):
    tests = {}
    for t in ("test_indomain", "test_crossdomain"):
        df = pd.read_csv(cfg.PROCESSED_DIR / f"{t}.csv")
        df["clean"] = clean_series(df["text"], aggressive=aggressive)
        tests[t.replace("test_", "")] = df
    return tests


def run_traditional(tests):
    rows = []
    for label, fname in SWEEP_POINTS:
        train = pd.read_csv(cfg.PROCESSED_DIR / f"train_{fname}.csv")
        train["clean"] = clean_series(train["text"])
        vec = TfidfVectorizer(ngram_range=cfg.TFIDF_NGRAM_RANGE, max_features=cfg.TFIDF_MAX_FEATURES)
        X = vec.fit_transform(train["clean"])
        y = train["label"].values

        lr = LogisticRegression(solver=cfg.LR_SOLVER, max_iter=1000).fit(X, y)
        svm = CalibratedClassifierCV(LinearSVC(), cv=3, method="isotonic").fit(X, y)

        for test_label, test_df in tests.items():
            Xt = vec.transform(test_df["clean"])
            yt = test_df["label"].values
            for mname, model in (("LR", lr), ("SVM", svm)):
                prob = model.predict_proba(Xt)[:, 1]
                pred = model.predict(Xt)
                m = compute_metrics(yt, pred, prob)
                rows.append({"sweep": label, "model": mname, "test": test_label,
                             "f1": round(m["f1"], 4), "precision": round(m["precision"], 4),
                             "recall": round(m["recall"], 4), "auc_roc": round(m["auc_roc"], 4)})
        print(f"  [LR/SVM] {label} ({fname}) done")
    return rows


# ---- CNN -- Vocab/CNNDataset/TextCNN/load_glove shared with train.py ----
from train import Vocab, CNNDataset, TextCNN, load_glove, get_cnn_vocab_and_embed


def run_cnn(tests):
    rows = []
    # Vocab built once from train_real_real (cheap to re-fetch -- GloVe itself
    # is cached inside get_cnn_vocab_and_embed, only the vocab lookup table is
    # actually reused across sweep points here).
    vocab, _ = get_cnn_vocab_and_embed()

    for label, fname in SWEEP_POINTS:
        repro.set_determinism(cfg.SEED)
        tr = pd.read_csv(cfg.PROCESSED_DIR / f"train_{fname}.csv")
        tr["clean"] = clean_series(tr["text"])
        gen = torch.Generator().manual_seed(cfg.SEED)
        dl = DataLoader(CNNDataset(tr["clean"], tr["label"], vocab, cfg.CNN_MAX_LEN),
                         batch_size=cfg.CNN_BATCH_SIZE, shuffle=True, generator=gen)
        # Fresh embed per sweep point -- see train.py's get_cnn_vocab_and_embed().
        _, embed = get_cnn_vocab_and_embed()
        model = TextCNN(embed, cfg.CNN_NUM_FILTERS, cfg.CNN_FILTER_SIZES, cfg.CNN_DROPOUT).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.CNN_LR)
        crit = torch.nn.CrossEntropyLoss()
        model.train()
        for _ in range(cfg.CNN_EPOCHS):
            for xb, yb in dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()

        # Save checkpoints for the two sweep points that don't already exist
        # elsewhere (swap_000/050/100 are the same data as real_real/mixed/
        # real_syn, already checkpointed by train.py) -- lets evaluate.py
        # hard-examples inspect what CNN gets wrong as synthetic fraction
        # rises, without retraining.
        if label in ("swap_025", "swap_075"):
            torch.save(model.state_dict(), cfg.MODELS_DIR / f"cnn_{label}.pt")

        model.eval()
        for test_label, test_df in tests.items():
            tdl = DataLoader(CNNDataset(test_df["clean"], test_df["label"], vocab, cfg.CNN_MAX_LEN),
                              batch_size=cfg.CNN_BATCH_SIZE)
            probs, preds, ys = [], [], []
            with torch.no_grad():
                for xb, yb in tdl:
                    logits = model(xb.to(DEVICE))
                    p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    probs.extend(p); preds.extend(logits.argmax(1).cpu().numpy()); ys.extend(yb.numpy())
            m = compute_metrics(np.array(ys), np.array(preds), np.array(probs))
            rows.append({"sweep": label, "model": "CNN", "test": test_label,
                         "f1": round(m["f1"], 4), "precision": round(m["precision"], 4),
                         "recall": round(m["recall"], 4), "auc_roc": round(m["auc_roc"], 4)})
        print(f"  [CNN] {label} ({fname}) done")
    return rows


# ---- BERT -- BertDataset shared with train.py ----
from train import BertDataset


def run_bert(tests, grad_accum):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    tok = AutoTokenizer.from_pretrained(cfg.BERT_MODEL_NAME)
    rows = []
    for label, fname in SWEEP_POINTS:
        repro.set_determinism(cfg.SEED)
        tr = pd.read_csv(cfg.PROCESSED_DIR / f"train_{fname}.csv")
        tr["clean"] = clean_series(tr["text"], aggressive=False)
        gen = torch.Generator().manual_seed(cfg.SEED)
        dl = DataLoader(BertDataset(tr["clean"], tr["label"], tok, cfg.BERT_MAX_LEN),
                         batch_size=cfg.BERT_BATCH_SIZE, shuffle=True, generator=gen)
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.BERT_MODEL_NAME, num_labels=2).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.BERT_LR)
        scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))
        # Linear warmup (10%) + decay, and gradient clipping -- see the
        # matching comment in train.py's train_bert().
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
        for test_label, test_df in tests.items():
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
            rows.append({"sweep": label, "model": "BERT", "test": test_label,
                         "f1": round(m["f1"], 4), "precision": round(m["precision"], 4),
                         "recall": round(m["recall"], 4), "auc_roc": round(m["auc_roc"], 4)})
        print(f"  [BERT] {label} ({fname}) done")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", choices=["lr_svm", "cnn", "bert"],
                     default=["lr_svm", "cnn", "bert"])
    ap.add_argument("--grad_accum", type=int, default=1)
    args = ap.parse_args()

    print(f"Device: {DEVICE}")
    all_rows = []
    if "lr_svm" in args.models:
        all_rows += run_traditional(load_tests(aggressive=True))
    if "cnn" in args.models:
        all_rows += run_cnn(load_tests(aggressive=True))
    if "bert" in args.models:
        all_rows += run_bert(load_tests(aggressive=False), args.grad_accum)

    df = pd.DataFrame(all_rows)
    out = EXTRA_DIR / "swap_sweep_results.csv"
    if out.exists():
        old = pd.read_csv(out)
        df = pd.concat([old, df], ignore_index=True).drop_duplicates(
            subset=["sweep", "model", "test"], keep="last")
    df.to_csv(out, index=False)
    print("\n=== SWAP SWEEP RESULTS ===")
    print(df.sort_values(["model", "test", "sweep"]).to_string(index=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
