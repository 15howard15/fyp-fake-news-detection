
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config as cfg
from preprocessing import clean_series
from metrics import compute_metrics

torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EXTRA_DIR = cfg.RESULTS_DIR / "extra"
EXTRA_DIR.mkdir(parents=True, exist_ok=True)

# The train sets to compare (only those that exist are run)
PAIRS = ["train_real_real", "train_mixed", "train_real_syn",
         "train_augmented", "train_lowres_real", "train_lowres_aug",
         "train_c2_synreal_realfake", "train_c3_synreal_synfake",
         "train_c6_full_augmented"]


# ======================================================================
# Shared: load the two test sets once
# ======================================================================
def load_tests(aggressive):
    tests = {}
    for t in ("test_indomain", "test_crossdomain"):
        path = cfg.PROCESSED_DIR / f"{t}.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["clean"] = clean_series(df["text"], aggressive=aggressive)
            tests[t.replace("test_", "")] = df
    if not tests:  # fallback
        df = pd.read_csv(cfg.PROCESSED_DIR / "test_shared.csv")
        df["clean"] = clean_series(df["text"], aggressive=aggressive)
        tests["crossdomain"] = df
    return tests


# ======================================================================
# CNN -- Vocab/CNNDataset/TextCNN/load_glove shared with train.py (were
# identical copy-pasted code). run_cnn() below keeps its OWN vocab policy
# (built from the largest train set here, not always train_real_real) --
# that's a deliberate difference from train.py, not duplication, so it
# stays local.
# ======================================================================
from train import Vocab, CNNDataset, TextCNN, load_glove


def run_cnn(train_sets, tests):
    rows = []
    # Build ONE vocab/embedding from the largest train set for fair comparison
    biggest = max(
        (t for t in train_sets if (cfg.PROCESSED_DIR / f"{t}.csv").exists()),
        key=lambda t: len(pd.read_csv(cfg.PROCESSED_DIR / f"{t}.csv")),
        default=None,
    )
    if biggest is None:
        print("  no CNN train sets found."); return rows
    base_txt = clean_series(pd.read_csv(cfg.PROCESSED_DIR / f"{biggest}.csv")["text"])
    vocab = Vocab(base_txt, cfg.CNN_VOCAB_SIZE)
    embed = load_glove(vocab, cfg.CNN_EMBED_DIM)

    for ts in train_sets:
        if not (cfg.PROCESSED_DIR / f"{ts}.csv").exists():
            continue
        # Reseed before EACH composition's training. Without this, the RNG
        # state a composition starts from depends on how many random draws
        # happened during every composition trained before it in this same
        # script run -- so results silently shift whenever train_sets is
        # reordered or extended (this is why re-running with mixed/real_syn
        # added earlier in the list changed augmented/lowres_aug's numbers).
        torch.manual_seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        print(f"[CNN] training on {ts} ...")
        tr = pd.read_csv(cfg.PROCESSED_DIR / f"{ts}.csv")
        tr["clean"] = clean_series(tr["text"])
        gen = torch.Generator().manual_seed(cfg.SEED)
        dl = DataLoader(CNNDataset(tr["clean"], tr["label"], vocab, cfg.CNN_MAX_LEN),
                        batch_size=cfg.CNN_BATCH_SIZE, shuffle=True, generator=gen)
        # .clone() -- see the comment in train.py's get_cnn_vocab_and_embed():
        # from_pretrained(freeze=False) shares memory with the tensor you pass
        # it, so reusing `embed` across train_sets without cloning would let
        # each training run's fine-tuning leak into the next one's "fresh"
        # starting point.
        model = TextCNN(embed.clone(), cfg.CNN_NUM_FILTERS, cfg.CNN_FILTER_SIZES, cfg.CNN_DROPOUT).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.CNN_LR)
        crit = nn.CrossEntropyLoss()
        model.train()
        for _ in range(cfg.CNN_EPOCHS):
            for xb, yb in dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()

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
            rows.append({"train_set": ts, "model": "CNN", "test": test_label,
                         "accuracy": round(m["accuracy"], 4), "precision": round(m["precision"], 4),
                         "recall": round(m["recall"], 4),
                         "f1": round(m["f1"], 4), "auc_roc": round(m["auc_roc"], 4)})
    return rows


# ======================================================================
# BERT -- BertDataset shared with train.py (identical code).
# ======================================================================
from train import BertDataset


def run_bert(train_sets, tests, grad_accum):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    tok = AutoTokenizer.from_pretrained(cfg.BERT_MODEL_NAME)
    rows = []
    for ts in train_sets:
        if not (cfg.PROCESSED_DIR / f"{ts}.csv").exists():
            continue
        # See the matching comment in run_cnn() -- reseed per composition so
        # results don't depend on train_sets ordering/length.
        torch.manual_seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        print(f"[BERT] training on {ts} ...")
        tr = pd.read_csv(cfg.PROCESSED_DIR / f"{ts}.csv")
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
            rows.append({"train_set": ts, "model": "BERT", "test": test_label,
                         "accuracy": round(m["accuracy"], 4), "precision": round(m["precision"], 4),
                         "recall": round(m["recall"], 4),
                         "f1": round(m["f1"], 4), "auc_roc": round(m["auc_roc"], 4)})
    return rows


def verdict(df, model):
    """Print the Option 1 augmentation verdict for a given model, cross-domain."""
    try:
        b = df[(df.train_set == "train_real_real") & (df.model == model) & (df.test == "crossdomain")]["f1"].values[0]
        a = df[(df.train_set == "train_augmented") & (df.model == model) & (df.test == "crossdomain")]["f1"].values[0]
        v = "HELPS" if a > b else "does not help"
        print(f"  [{model}] real_real F1={b}  vs  augmented F1={a}  ->  synthetic {v}")
    except (IndexError, KeyError):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", choices=["cnn", "bert"], default=["cnn"])
    ap.add_argument("--grad_accum", type=int, default=1)
    args = ap.parse_args()

    print(f"Device: {DEVICE}")
    all_rows = []
    if "cnn" in args.models:
        all_rows += run_cnn(PAIRS, load_tests(aggressive=True))
    if "bert" in args.models:
        all_rows += run_bert(PAIRS, load_tests(aggressive=False), args.grad_accum)

    df = pd.DataFrame(all_rows)
    out = EXTRA_DIR / "deep_augmentation_results.csv"
    # append if file exists so cnn + bert runs accumulate
    if out.exists():
        df = pd.concat([pd.read_csv(out), df], ignore_index=True).drop_duplicates(
            subset=["train_set", "model", "test"], keep="last")
    df.to_csv(out, index=False)

    print("\n=== DEEP RESULTS ===")
    print(df.to_string(index=False))
    print(f"\nSaved to {out}")
    print("\n--- Option 1 verdict (cross-domain F1) ---")
    for mdl in ("CNN", "BERT"):
        verdict(df, mdl)


if __name__ == "__main__":
    main()
