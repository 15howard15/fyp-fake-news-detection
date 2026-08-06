"""
evaluate.py -- unified evaluation entry point.

Merges 08_evaluate.py and 11_error_analysis.py (deleted after this file was
verified to reproduce their output exactly) into one script with
subcommands. Unlike train.py's merge, these two didn't share
much actual CODE -- 08 just aggregates already-saved metrics_*.json files,
while 11 runs its own fresh predictions to capture confusion matrices -- so
this is a file-count consolidation, not a duplication removal.
eval_style_robustness.py stays separate: it's a structurally different
analysis (paired attack-vs-original evaluation of already-trained models),
not metric aggregation, so folding it in here wouldn't actually simplify
anything.

Examples:
    python evaluate.py master                     # aggregate metrics_*.json -> master_results.csv + f1_heatmap.png
    python evaluate.py error-analysis              # LR+SVM confusion/prediction-distribution + vocab overlap (seconds)
    python evaluate.py error-analysis --deep       # + CNN + BERT (needs GPU)
    python evaluate.py error-analysis --deep --models cnn   # pick which deep models
    python evaluate.py cross-target --dataset welfake      # evaluate real_real/mixed/real_syn on test_crossdomain2 (WELFake)
    python evaluate.py cross-target --dataset welfake --comp c2_synreal_realfake c3_synreal_synfake
    python evaluate.py seed-summary                # mean +/- std across seeds, from multiseed_results.csv
    python evaluate.py case-studies                       # concrete style-attack flip examples, BERT, mixed vs style_robust
    python evaluate.py case-studies --model lr --comp real_syn --n 5
"""
import argparse
import json
import sys

# Windows' default console codepage (cp1252/gbk, not UTF-8) can't print
# certain characters that show up in real news text (curly quotes, em
# dashes, etc.) -- hard-examples/case-studies print raw article text, and a
# single un-encodable character used to crash the whole run AFTER inference
# had already finished but BEFORE any results were saved (the CSV write
# happens once at the end of the loop). Reconfigure stdout to replace
# un-encodable characters instead of raising, so a print never loses work.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

import config as cfg
from preprocessing import clean_series
from metrics import compute_metrics

EXTRA_DIR = cfg.RESULTS_DIR / "extra"
EXTRA_DIR.mkdir(parents=True, exist_ok=True)

# The 5 conditions in the dashboard's merged "Replacement (full matrix)" tab.
ERROR_ANALYSIS_COMPS = ["real_real", "mixed", "real_syn", "c2_synreal_realfake", "c3_synreal_synfake"]


# ======================================================================
# `master` -- aggregate metrics_*.json into master_results.csv + heatmap
# ======================================================================
def cmd_master(args):
    rows = []
    for path in sorted(cfg.RESULTS_DIR.glob("metrics_*.json")):
        with open(path) as f:
            d = json.load(f)
        rows.append({
            "model": d["model"],
            "composition": d["composition"],
            "accuracy": round(d["accuracy"], 4),
            "precision": round(d["precision"], 4),
            "recall": round(d["recall"], 4),
            "f1": round(d["f1"], 4),
            "auc_roc": round(d.get("auc_roc", float("nan")), 4),
        })

    if not rows:
        print("No metrics found. Run the training scripts first.")
        return

    df = pd.DataFrame(rows)
    # This file is the Q1 replacement-matrix source (real_real/mixed/real_syn
    # only) -- results/ also holds metrics for compositions added later
    # (style_robust, real_syn_multisource, seed-repeat runs, etc.) that don't
    # belong in this table. Filter to cfg.COMPOSITIONS BEFORE building the
    # pivot: Categorical-with-fixed-categories alone doesn't drop non-matching
    # rows, it just turns them into NaN composition values, which then
    # collide during pivot() (>1 row per (model, NaN)) and crash the heatmap.
    df = df[df["composition"].isin(cfg.COMPOSITIONS)].reset_index(drop=True)

    # order nicely
    model_order = ["LR", "SVM", "CNN", "BERT"]
    df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
    df["composition"] = pd.Categorical(df["composition"],
                                       categories=cfg.COMPOSITIONS, ordered=True)
    df = df.sort_values(["model", "composition"]).reset_index(drop=True)

    out_csv = cfg.RESULTS_DIR / "master_results.csv"
    df.to_csv(out_csv, index=False)
    print("\n=== MASTER RESULTS ===")
    print(df.to_string(index=False))
    print(f"\nSaved to {out_csv}")

    # F1 heatmap (model x composition)
    pivot = df.pivot(index="model", columns="composition", values="f1")
    plt.figure(figsize=(7, 4))
    plt.imshow(pivot.values, aspect="auto", cmap="viridis")
    plt.colorbar(label="F1")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if pd.notna(val):
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", color="white")
    plt.title("F1 by model and data composition")
    plt.tight_layout()
    plt.savefig(cfg.RESULTS_DIR / "f1_heatmap.png", dpi=150)
    plt.close()
    print(f"Heatmap saved to {cfg.RESULTS_DIR / 'f1_heatmap.png'}")


# ======================================================================
# `error-analysis` -- WHY does train_real_syn collapse to ~0 F1 cross-domain?
#
# Two pieces of evidence, both cheap to compute from what's already built:
#
#   (A) Prediction distribution + confusion matrix for the 5 "replacement
#       axis" compositions (real_real, mixed, real_syn, C2, C3), all 4
#       models, on test_crossdomain. compute_metrics() already returns a
#       confusion matrix (see metrics.py) but 09/10 never saved it to the
#       results CSV -- this captures it so we can see WHETHER a model is
#       predicting almost everything as one class (as opposed to just being
#       "less accurate").
#
#   (B) Vocabulary/length overlap: how much of test_crossdomain's fake-class
#       vocabulary is unseen relative to train_real_real, compared to
#       test_indomain's fake class. Turns the "LIAR is a genre/length shift,
#       not just a topic shift" caveat into a measured number instead of a
#       hand-wave.
# ======================================================================
def pred_dist(preds, y_true=None):
    preds = np.asarray(preds)
    n = len(preds)
    d = {
        "pred_real_pct": round(100 * (preds == cfg.LABEL_REAL).sum() / n, 1),
        "pred_fake_pct": round(100 * (preds == cfg.LABEL_FAKE).sum() / n, 1),
    }
    if y_true is not None:
        y_true = np.asarray(y_true)
        d["true_real_pct"] = round(100 * (y_true == cfg.LABEL_REAL).sum() / n, 1)
        d["true_fake_pct"] = round(100 * (y_true == cfg.LABEL_FAKE).sum() / n, 1)
    return d


def run_traditional(test_df):
    rows = []
    for comp in ERROR_ANALYSIS_COMPS:
        path = cfg.PROCESSED_DIR / f"train_{comp}.csv"
        if not path.exists():
            print(f"  (skip {comp} — not built)")
            continue
        train = pd.read_csv(path)
        train["clean"] = clean_series(train["text"])
        vec = TfidfVectorizer(ngram_range=cfg.TFIDF_NGRAM_RANGE, max_features=cfg.TFIDF_MAX_FEATURES)
        X_train = vec.fit_transform(train["clean"])
        y_train = train["label"].values
        Xt = vec.transform(test_df["clean"])
        yt = test_df["label"].values

        lr = LogisticRegression(solver=cfg.LR_SOLVER, max_iter=1000).fit(X_train, y_train)
        # method="isotonic" -- see train.py's train_lr_svm(). Default "sigmoid"
        # calibration can invert the ranking when LinearSVC's scores are
        # near-perfectly separable, which is common for TF-IDF text features.
        svm = CalibratedClassifierCV(LinearSVC(), cv=3, method="isotonic").fit(X_train, y_train)
        for mname, model in (("LR", lr), ("SVM", svm)):
            prob = model.predict_proba(Xt)[:, 1]
            pred = model.predict(Xt)
            m = compute_metrics(yt, pred, prob)
            row = {"train_set": comp, "model": mname, "f1": round(m["f1"], 4),
                   **m["confusion"], **pred_dist(pred, yt)}
            rows.append(row)
            print(f"  [{mname:4s}|{comp:24s}] f1={row['f1']:.3f}  "
                  f"pred: real={row['pred_real_pct']:5.1f}%  fake={row['pred_fake_pct']:5.1f}%  "
                  f"(true: real={row['true_real_pct']:.1f}% fake={row['true_fake_pct']:.1f}%)  "
                  f"confusion(tn,fp,fn,tp)=({row['tn']},{row['fp']},{row['fn']},{row['tp']})")
    return rows


def run_deep(test_df, models):
    """Mirrors run_deep_extra_experiments.py's run_cnn/run_bert but scoped to
    ERROR_ANALYSIS_COMPS and capturing the confusion matrix + prediction
    distribution."""
    import torch
    from torch.utils.data import DataLoader
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []

    if "cnn" in models:
        # Vocab/CNNDataset/TextCNN/load_glove shared with train.py (were
        # identical copy-pasted code). Fixed vocab from train_real_real so
        # all 5 CNNs here are on equal footing -- same policy train.py uses.
        import torch.nn as nn
        from train import Vocab, CNNDataset, TextCNN, load_glove, get_cnn_vocab_and_embed

        test_clean = clean_series(test_df["text"])
        vocab, _ = get_cnn_vocab_and_embed()
        tdl = DataLoader(CNNDataset(test_clean, test_df["label"], vocab, cfg.CNN_MAX_LEN),
                          batch_size=cfg.CNN_BATCH_SIZE)

        for comp in ERROR_ANALYSIS_COMPS:
            path = cfg.PROCESSED_DIR / f"train_{comp}.csv"
            if not path.exists():
                continue
            torch.manual_seed(cfg.SEED)
            np.random.seed(cfg.SEED)
            print(f"[CNN] training on {comp} ...")
            tr = pd.read_csv(path)
            tr["clean"] = clean_series(tr["text"])
            gen = torch.Generator().manual_seed(cfg.SEED)
            dl = DataLoader(CNNDataset(tr["clean"], tr["label"], vocab, cfg.CNN_MAX_LEN),
                             batch_size=cfg.CNN_BATCH_SIZE, shuffle=True, generator=gen)
            # Fresh embed per composition -- see train.py's get_cnn_vocab_and_embed().
            _, embed = get_cnn_vocab_and_embed()
            model = TextCNN(embed, cfg.CNN_NUM_FILTERS, cfg.CNN_FILTER_SIZES, cfg.CNN_DROPOUT).to(DEVICE)
            opt = torch.optim.Adam(model.parameters(), lr=cfg.CNN_LR)
            crit = nn.CrossEntropyLoss()
            model.train()
            for _ in range(cfg.CNN_EPOCHS):
                for xb, yb in dl:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()

            model.eval()
            probs, preds, ys = [], [], []
            with torch.no_grad():
                for xb, yb in tdl:
                    logits = model(xb.to(DEVICE))
                    p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    probs.extend(p); preds.extend(logits.argmax(1).cpu().numpy()); ys.extend(yb.numpy())
            m = compute_metrics(np.array(ys), np.array(preds), np.array(probs))
            row = {"train_set": comp, "model": "CNN", "f1": round(m["f1"], 4),
                   **m["confusion"], **pred_dist(preds, ys)}
            rows.append(row)
            print(f"  [CNN |{comp:24s}] f1={row['f1']:.3f}  "
                  f"pred: real={row['pred_real_pct']:5.1f}%  fake={row['pred_fake_pct']:5.1f}%  "
                  f"confusion(tn,fp,fn,tp)=({row['tn']},{row['fp']},{row['fn']},{row['tp']})")

    if "bert" in models:
        from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                                   get_linear_schedule_with_warmup)
        # BertDataset shared with train.py (identical code).
        from train import BertDataset

        tok = AutoTokenizer.from_pretrained(cfg.BERT_MODEL_NAME)
        test_clean = clean_series(test_df["text"], aggressive=False)
        tdl = DataLoader(BertDataset(test_clean, test_df["label"], tok, cfg.BERT_MAX_LEN),
                          batch_size=cfg.BERT_BATCH_SIZE)

        for comp in ERROR_ANALYSIS_COMPS:
            path = cfg.PROCESSED_DIR / f"train_{comp}.csv"
            if not path.exists():
                continue
            torch.manual_seed(cfg.SEED)
            np.random.seed(cfg.SEED)
            print(f"[BERT] training on {comp} ...")
            tr = pd.read_csv(path)
            tr["clean"] = clean_series(tr["text"], aggressive=False)
            gen = torch.Generator().manual_seed(cfg.SEED)
            dl = DataLoader(BertDataset(tr["clean"], tr["label"], tok, cfg.BERT_MAX_LEN),
                             batch_size=cfg.BERT_BATCH_SIZE, shuffle=True, generator=gen)
            model = AutoModelForSequenceClassification.from_pretrained(
                cfg.BERT_MODEL_NAME, num_labels=2).to(DEVICE)
            opt = torch.optim.AdamW(model.parameters(), lr=cfg.BERT_LR)
            scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))
            # Linear warmup (10%) + decay, and gradient clipping -- matches
            # train.py's train_bert() recipe.
            total_steps = len(dl) * cfg.BERT_EPOCHS
            warmup_steps = max(1, int(0.1 * total_steps))
            sched = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)
            model.train()
            for _ in range(cfg.BERT_EPOCHS):
                opt.zero_grad()
                for step, batch in enumerate(dl):
                    batch = {k: v.to(DEVICE) for k, v in batch.items()}
                    with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                        loss = model(**batch).loss
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()

            model.eval()
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
            row = {"train_set": comp, "model": "BERT", "f1": round(m["f1"], 4),
                   **m["confusion"], **pred_dist(preds, ys)}
            rows.append(row)
            print(f"  [BERT|{comp:24s}] f1={row['f1']:.3f}  "
                  f"pred: real={row['pred_real_pct']:5.1f}%  fake={row['pred_fake_pct']:5.1f}%  "
                  f"confusion(tn,fp,fn,tp)=({row['tn']},{row['fp']},{row['fn']},{row['tp']})")

    return rows


def vocab_overlap_report():
    train = pd.read_csv(cfg.PROCESSED_DIR / "train_real_real.csv")
    indomain = pd.read_csv(cfg.PROCESSED_DIR / "test_indomain.csv")
    crossdomain = pd.read_csv(cfg.PROCESSED_DIR / "test_crossdomain.csv")

    train_clean = clean_series(train["text"])
    vec = TfidfVectorizer(ngram_range=(1, 1), max_features=cfg.TFIDF_MAX_FEATURES)
    vec.fit(train_clean)
    train_vocab = set(vec.vocabulary_.keys())

    def oov_and_length(df, label, name):
        sub = df[df["label"] == label]
        clean = clean_series(sub["text"])
        all_tokens, oov_tokens, lengths = 0, 0, []
        for t in clean:
            toks = t.split()
            lengths.append(len(toks))
            all_tokens += len(toks)
            oov_tokens += sum(1 for w in toks if w not in train_vocab)
        oov_rate = round(100 * oov_tokens / max(all_tokens, 1), 1)
        avg_len = round(float(np.mean(lengths)), 1) if lengths else 0.0
        print(f"  {name:28s} n={len(sub):5d}  avg_words={avg_len:6.1f}  oov_rate={oov_rate:5.1f}%")
        return {"set": name, "n": len(sub), "avg_words": avg_len, "oov_rate_pct": oov_rate}

    print("\n--- Vocabulary overlap vs. train_real_real (fitted on aggressive-cleaned text) ---")
    print(f"  train_real_real vocab size: {len(train_vocab)}")
    rep = [
        oov_and_length(indomain, cfg.LABEL_REAL, "test_indomain (real, ISOT held-out)"),
        oov_and_length(indomain, cfg.LABEL_FAKE, "test_indomain (fake, ISOT held-out)"),
        oov_and_length(crossdomain, cfg.LABEL_REAL, "test_crossdomain (real, ISOT held-out)"),
        oov_and_length(crossdomain, cfg.LABEL_FAKE, "test_crossdomain (fake, LIAR)"),
    ]
    return rep


def cmd_error_analysis(args):
    test_df = pd.read_csv(cfg.PROCESSED_DIR / "test_crossdomain.csv")
    test_df["clean"] = clean_series(test_df["text"])

    print("=== (A) Confusion matrix / prediction distribution, test_crossdomain ===")
    rows = run_traditional(test_df)
    if args.deep:
        rows += run_deep(test_df, args.models)

    df = pd.DataFrame(rows)
    out_a = EXTRA_DIR / "error_analysis_confusion.csv"
    if out_a.exists() and not df.empty:
        old = pd.read_csv(out_a)
        df = pd.concat([old, df], ignore_index=True).drop_duplicates(
            subset=["train_set", "model"], keep="last")
    if not df.empty:
        df.to_csv(out_a, index=False)
        print(f"\nSaved to {out_a}")

    vocab_rows = vocab_overlap_report()
    out_b = EXTRA_DIR / "vocab_overlap.json"
    with open(out_b, "w") as f:
        json.dump(vocab_rows, f, indent=2)
    print(f"Saved to {out_b}")


CROSS_TARGET_ALIASES = {"welfake": "test_crossdomain2", "liar": "test_crossdomain"}


def cmd_cross_target(args):
    """Evaluate the already-trained real_real/mixed/real_syn checkpoints
    (no retraining -- a model's weights don't depend on which test set you
    later evaluate on) against a cross-domain test set OTHER than the
    default test_crossdomain.csv (LIAR-based). Point this at
    test_crossdomain2.csv (WELFake-based, built by 04c_build_indomain_
    test.py) to compare generalization across two different cross-domain
    targets instead of relying on just one, which conflates domain shift
    with LIAR's genre/length shift (see README)."""
    test_name = CROSS_TARGET_ALIASES.get(args.dataset, args.dataset)
    test_path = cfg.PROCESSED_DIR / f"{test_name}.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"{test_path} not found. Run build_test_sets.py first.")
    test = pd.read_csv(test_path)
    y = test["label"].values

    rows = []
    for comp in args.comp:
        # ---- LR / SVM ----
        try:
            vec = joblib.load(cfg.MODELS_DIR / f"tfidf_{comp}.joblib")
            Xt = vec.transform(clean_series(test["text"]))
            for mname, fname in (("LR", "lr"), ("SVM", "svm")):
                clf = joblib.load(cfg.MODELS_DIR / f"{fname}_{comp}.joblib")
                pred = clf.predict(Xt)
                prob = clf.predict_proba(Xt)[:, 1]
                m = compute_metrics(y, pred, prob)
                rows.append({"dataset": args.dataset, "comp": comp, "model": mname,
                             "accuracy": round(m["accuracy"], 4), "precision": round(m["precision"], 4),
                             "recall": round(m["recall"], 4), "f1": round(m["f1"], 4),
                             "auc_roc": round(m["auc_roc"], 4)})
                print(f"  [{mname:4s}|{comp:10s}] f1={m['f1']:.4f}  auc={m['auc_roc']:.4f}")
        except FileNotFoundError as e:
            print(f"  (skip LR/SVM for {comp} -- {e})")

        # ---- CNN ----
        try:
            import torch
            from torch.utils.data import DataLoader
            from train import CNNDataset, TextCNN, get_cnn_vocab_and_embed
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            vocab, embed = get_cnn_vocab_and_embed()
            model = TextCNN(embed, cfg.CNN_NUM_FILTERS, cfg.CNN_FILTER_SIZES, cfg.CNN_DROPOUT).to(DEVICE)
            model.load_state_dict(torch.load(cfg.MODELS_DIR / f"cnn_{comp}.pt", map_location=DEVICE))
            model.eval()
            test_clean = clean_series(test["text"])
            dl = DataLoader(CNNDataset(test_clean, y, vocab, cfg.CNN_MAX_LEN), batch_size=cfg.CNN_BATCH_SIZE)
            probs, preds = [], []
            with torch.no_grad():
                for xb, _ in dl:
                    logits = model(xb.to(DEVICE))
                    probs.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy())
                    preds.extend(logits.argmax(1).cpu().numpy())
            m = compute_metrics(y, np.array(preds), np.array(probs))
            rows.append({"dataset": args.dataset, "comp": comp, "model": "CNN",
                         "accuracy": round(m["accuracy"], 4), "precision": round(m["precision"], 4),
                         "recall": round(m["recall"], 4), "f1": round(m["f1"], 4),
                         "auc_roc": round(m["auc_roc"], 4)})
            print(f"  [CNN |{comp:10s}] f1={m['f1']:.4f}  auc={m['auc_roc']:.4f}")
        except FileNotFoundError as e:
            print(f"  (skip CNN for {comp} -- {e})")

        # ---- BERT ----
        try:
            import torch
            from torch.utils.data import DataLoader
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            from train import BertDataset
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            tok = AutoTokenizer.from_pretrained(cfg.BERT_MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(cfg.MODELS_DIR / f"bert_{comp}").to(DEVICE)
            model.eval()
            test_clean = clean_series(test["text"], aggressive=False)
            dl = DataLoader(BertDataset(test_clean, y, tok, cfg.BERT_MAX_LEN), batch_size=cfg.BERT_BATCH_SIZE)
            probs, preds = [], []
            with torch.no_grad():
                for batch in dl:
                    batch.pop("labels")
                    batch = {k: v.to(DEVICE) for k, v in batch.items()}
                    with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                        logits = model(**batch).logits
                    probs.extend(torch.softmax(logits.float(), 1)[:, 1].cpu().numpy())
                    preds.extend(logits.argmax(1).cpu().numpy())
            m = compute_metrics(y, np.array(preds), np.array(probs))
            rows.append({"dataset": args.dataset, "comp": comp, "model": "BERT",
                         "accuracy": round(m["accuracy"], 4), "precision": round(m["precision"], 4),
                         "recall": round(m["recall"], 4), "f1": round(m["f1"], 4),
                         "auc_roc": round(m["auc_roc"], 4)})
            print(f"  [BERT|{comp:10s}] f1={m['f1']:.4f}  auc={m['auc_roc']:.4f}")
        except (FileNotFoundError, OSError) as e:
            print(f"  (skip BERT for {comp} -- {e})")

    df = pd.DataFrame(rows)
    out = EXTRA_DIR / "crossdomain2_results.csv"
    df.to_csv(out, index=False)
    print(f"\n=== CROSS-TARGET RESULTS ({args.dataset}) ===")
    print(df.to_string(index=False))
    print(f"\nSaved to {out}")


def cmd_hard_examples(args):
    """Find the most confidently-WRONG predictions for an already-trained
    checkpoint on a plain test set (test_crossdomain.csv / test_crossdomain2.csv)
    -- the concrete misclassified examples behind an aggregate F1/AUC number.
    No retraining: loads the saved checkpoint and runs inference only, same
    loading pattern as cmd_cross_target. 'Confidently wrong' = sorted by the
    probability the model assigned to its own (incorrect) prediction, so the
    first examples shown are the ones the model was most sure about and most
    wrong on -- the most informative cases for explaining a failure, not just
    any wrong prediction."""
    test_name = CROSS_TARGET_ALIASES.get(args.dataset, args.dataset)
    test_path = cfg.PROCESSED_DIR / f"{test_name}.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"{test_path} not found. Run build_test_sets.py first.")
    test = pd.read_csv(test_path).reset_index(drop=True)
    y = test["label"].values

    def predict_lr_svm(comp):
        fname = "lr" if args.model == "lr" else "svm"
        vec = joblib.load(cfg.MODELS_DIR / f"tfidf_{comp}.joblib")
        clf = joblib.load(cfg.MODELS_DIR / f"{fname}_{comp}.joblib")
        Xt = vec.transform(clean_series(test["text"]))
        return clf.predict(Xt), clf.predict_proba(Xt)[:, 1]

    def predict_cnn(comp):
        import torch
        from torch.utils.data import DataLoader
        from train import CNNDataset, TextCNN, get_cnn_vocab_and_embed
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        vocab, embed = get_cnn_vocab_and_embed()
        model = TextCNN(embed, cfg.CNN_NUM_FILTERS, cfg.CNN_FILTER_SIZES, cfg.CNN_DROPOUT).to(DEVICE)
        model.load_state_dict(torch.load(cfg.MODELS_DIR / f"cnn_{comp}.pt", map_location=DEVICE))
        model.eval()
        dl = DataLoader(CNNDataset(clean_series(test["text"]), y, vocab, cfg.CNN_MAX_LEN),
                         batch_size=cfg.CNN_BATCH_SIZE)
        probs, preds = [], []
        with torch.no_grad():
            for xb, _ in dl:
                logits = model(xb.to(DEVICE))
                probs.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy())
                preds.extend(logits.argmax(1).cpu().numpy())
        return np.array(preds), np.array(probs)

    def predict_bert(comp):
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from train import BertDataset
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(cfg.BERT_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(cfg.MODELS_DIR / f"bert_{comp}").to(DEVICE)
        model.eval()
        dl = DataLoader(BertDataset(clean_series(test["text"], aggressive=False), y, tok, cfg.BERT_MAX_LEN),
                         batch_size=cfg.BERT_BATCH_SIZE)
        probs, preds = [], []
        with torch.no_grad():
            for batch in dl:
                batch.pop("labels")
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                    logits = model(**batch).logits
                probs.extend(torch.softmax(logits.float(), 1)[:, 1].cpu().numpy())
                preds.extend(logits.argmax(1).cpu().numpy())
        return np.array(preds), np.array(probs)

    predict_fn = {"lr": predict_lr_svm, "svm": predict_lr_svm, "cnn": predict_cnn, "bert": predict_bert}[args.model]

    saved_rows = []
    for comp in args.comp:
        try:
            pred, prob = predict_fn(comp)
        except (FileNotFoundError, OSError) as e:
            print(f"  (skip {comp} -- {e})")
            continue
        wrong = pred != y
        # confidence of the WRONG prediction: P(fake) if it predicted fake, else P(real)
        wrong_conf = np.where(pred == cfg.LABEL_FAKE, prob, 1 - prob)
        order = [i for i in np.argsort(-wrong_conf) if wrong[i]][: args.n]
        print(f"\n=== {args.model.upper()} | {comp} | {args.dataset}: {int(wrong.sum())}/{len(y)} wrong ===")
        if not order:
            print("  (no wrong predictions found)")
        for i in order:
            true_name = "REAL" if y[i] == cfg.LABEL_REAL else "FAKE"
            pred_name = "REAL" if pred[i] == cfg.LABEL_REAL else "FAKE"
            src = test["source"].iloc[i] if "source" in test.columns else "?"
            print(f"\n  true={true_name}  predicted={pred_name} (confidence {wrong_conf[i]:.3f})  source={src}")
            print(f"    {str(test['text'].iloc[i])[:300]}...")
            saved_rows.append({
                "model": args.model, "comp": comp, "dataset": args.dataset,
                "true_label": true_name, "predicted_label": pred_name,
                "confidence": round(float(wrong_conf[i]), 4), "source": src,
                "text": test["text"].iloc[i],
            })

    if saved_rows:
        out = EXTRA_DIR / "hard_examples.csv"
        pd.DataFrame(saved_rows).to_csv(out, index=False)
        print(f"\nSaved {len(saved_rows)} examples to {out}")


def cmd_seed_summary(args):
    """Collapse results/extra/multiseed_results.csv (3 seeds x 5 compositions,
    CNN + BERT) into a mean +/- std table -- so the thesis can cite seed-
    averaged numbers for the two non-deterministic models instead of single-
    run point estimates."""
    path = EXTRA_DIR / "multiseed_results.csv"
    if not path.exists():
        print(f"{path} not found. Run run_multiseed_robustness.py first.")
        return
    df = pd.read_csv(path)

    rows = []
    for (model, comp, test), g in df.groupby(["model", "comp", "test"]):
        f1_mean, f1_std = g["f1"].mean(), g["f1"].std()
        auc_mean, auc_std = g["auc_roc"].mean(), g["auc_roc"].std()
        n = len(g)
        rows.append({
            "model": model, "comp": comp, "test": test, "n_seeds": n,
            "f1_mean": round(f1_mean, 4),
            "f1_std": round(f1_std, 4) if n > 1 and pd.notna(f1_std) else 0.0,
            "auc_mean": round(auc_mean, 4),
            "auc_std": round(auc_std, 4) if n > 1 and pd.notna(auc_std) else 0.0,
        })
    out_df = pd.DataFrame(rows).sort_values(["model", "comp", "test"]).reset_index(drop=True)
    out_df["f1_pretty"] = out_df.apply(
        lambda r: f"{r.f1_mean:.3f} +/- {r.f1_std:.3f}" if r.n_seeds > 1 else f"{r.f1_mean:.3f} (n=1)",
        axis=1,
    )

    print("\n=== SEED-AVERAGED SUMMARY (mean +/- std across seeds) ===")
    print(out_df.to_string(index=False))

    out_path = EXTRA_DIR / "seed_summary.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path} -- f1_pretty column is paste-ready for a thesis table.")


def cmd_case_studies(args):
    """Find concrete articles where the style attack flipped a prediction --
    the actual text behind style_robustness_results.csv's flip_rate numbers.
    No retraining: loads the already-saved LR/SVM/CNN/BERT checkpoints for
    each composition and re-runs inference on the paired original/attacked
    style-attack set (data/synthetic/style_attack_originals.csv + _attack.csv),
    same as eval_style_robustness.py's approach but keeping only the
    specific examples where a correct prediction became wrong."""
    orig = pd.read_csv(cfg.SYNTHETIC_DIR / "style_attack_originals.csv").sort_values("orig_id").reset_index(drop=True)
    attacked = pd.read_csv(cfg.SYNTHETIC_DIR / "style_attack.csv").sort_values("orig_id").reset_index(drop=True)
    assert (orig["orig_id"].values == attacked["orig_id"].values).all(), "misaligned pairs"
    y = orig["label"].values

    def predict_lr_svm(comp):
        fname = "lr" if args.model == "lr" else "svm"
        vec = joblib.load(cfg.MODELS_DIR / f"tfidf_{comp}.joblib")
        clf = joblib.load(cfg.MODELS_DIR / f"{fname}_{comp}.joblib")
        po = clf.predict(vec.transform(clean_series(orig["text"])))
        pa = clf.predict(vec.transform(clean_series(attacked["text"])))
        return po, pa

    def predict_cnn(comp):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from train import CNNDataset, get_cnn_vocab_and_embed
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        class _LoadableTextCNN(nn.Module):
            # Empty-shell constructor for load_state_dict -- NOT the same
            # signature as train.py's TextCNN (which needs a pretrained
            # embedding matrix to construct from). Mirrors 15_eval_style_
            # robustness.py's TextCNN, which stayed a separate file for
            # this exact reason: loading and training need different
            # constructors around the same architecture.
            def __init__(self, embed_dim, num_filters, filter_sizes, dropout, vocab_size):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, fs) for fs in filter_sizes])
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(num_filters * len(filter_sizes), 2)

            def forward(self, x):
                x = self.embedding(x).permute(0, 2, 1)
                pooled = [torch.relu(c(x)).max(dim=2).values for c in self.convs]
                return self.fc(self.dropout(torch.cat(pooled, dim=1)))

        vocab, _ = get_cnn_vocab_and_embed()
        model = _LoadableTextCNN(cfg.CNN_EMBED_DIM, cfg.CNN_NUM_FILTERS, cfg.CNN_FILTER_SIZES,
                                  cfg.CNN_DROPOUT, len(vocab)).to(DEVICE)
        model.load_state_dict(torch.load(cfg.MODELS_DIR / f"cnn_{comp}.pt", map_location=DEVICE))
        model.eval()

        def predict(texts):
            dl = DataLoader(CNNDataset(clean_series(texts), [0] * len(texts), vocab, cfg.CNN_MAX_LEN),
                             batch_size=cfg.CNN_BATCH_SIZE)
            preds = []
            with torch.no_grad():
                for xb, _ in dl:
                    preds.extend(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
            return np.array(preds)

        return predict(orig["text"]), predict(attacked["text"])

    def predict_bert(comp):
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from train import BertDataset
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(cfg.BERT_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(cfg.MODELS_DIR / f"bert_{comp}").to(DEVICE)
        model.eval()

        def predict(texts):
            dl = DataLoader(BertDataset(clean_series(texts, aggressive=False), [0] * len(texts), tok, cfg.BERT_MAX_LEN),
                             batch_size=cfg.BERT_BATCH_SIZE)
            preds = []
            with torch.no_grad():
                for batch in dl:
                    batch.pop("labels")
                    batch = {k: v.to(DEVICE) for k, v in batch.items()}
                    preds.extend(model(**batch).logits.argmax(1).cpu().numpy())
            return np.array(preds)

        return predict(orig["text"]), predict(attacked["text"])

    predict_fn = {"lr": predict_lr_svm, "svm": predict_lr_svm, "cnn": predict_cnn, "bert": predict_bert}[args.model]

    saved_rows = []
    for comp in args.comp:
        try:
            po, pa = predict_fn(comp)
        except (FileNotFoundError, OSError) as e:
            print(f"  (skip {comp} -- {e})")
            continue
        flipped = (po == y) & (pa != y)
        print(f"\n=== {args.model.upper()} | {comp}: {int(flipped.sum())}/{len(y)} correct-before -> wrong-after ===")
        idx = np.where(flipped)[0][:args.n]
        if len(idx) == 0:
            print("  (no flips found -- attack didn't fool this model/composition)")
        for i in idx:
            label_name = "REAL" if y[i] == cfg.LABEL_REAL else "FAKE"
            print(f"\n  [{orig['orig_id'].iloc[i]}] true={label_name}  attack={orig['attack_type'].iloc[i]}")
            print(f"    BEFORE (correct): {orig['text'].iloc[i][:300]}...")
            print(f"    AFTER  (wrong)  : {attacked['text'].iloc[i][:300]}...")
            saved_rows.append({
                "model": args.model, "comp": comp, "orig_id": orig["orig_id"].iloc[i],
                "true_label": label_name, "attack_type": orig["attack_type"].iloc[i],
                "text_before": orig["text"].iloc[i], "text_after": attacked["text"].iloc[i],
            })

    if saved_rows:
        out = EXTRA_DIR / "case_studies.csv"
        pd.DataFrame(saved_rows).to_csv(out, index=False)
        print(f"\nSaved {len(saved_rows)} full-text examples to {out}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("master", help="aggregate metrics_*.json into master_results.csv + f1_heatmap.png")

    ea = sub.add_parser(
        "error-analysis",
        help="confusion-matrix / prediction-distribution + vocab-overlap analysis (see module docstring)",
    )
    ea.add_argument("--deep", action="store_true", help="also run CNN/BERT (needs GPU)")
    ea.add_argument("--models", nargs="+", choices=["cnn", "bert"], default=["cnn", "bert"],
                     help="which deep models to include when --deep is set")

    ct = sub.add_parser(
        "cross-target",
        help="evaluate already-trained compositions on a different cross-domain test set (see module docstring)",
    )
    ct.add_argument("--dataset", default="welfake",
                     help="'welfake' (-> test_crossdomain2.csv) or a raw composition/test-set stem")
    ct.add_argument("--comp", nargs="+", default=list(cfg.COMPOSITIONS),
                     help="composition name(s) to evaluate (checkpoints must exist under models/)")

    sub.add_parser("seed-summary", help="mean +/- std across seeds, from results/extra/multiseed_results.csv")

    cs = sub.add_parser(
        "case-studies",
        help="concrete style-attack flip examples with article text (see module docstring)",
    )
    cs.add_argument("--model", choices=["lr", "svm", "cnn", "bert"], default="bert")
    cs.add_argument("--comp", nargs="+", default=["mixed", "style_robust"],
                     help="compositions to check (checkpoints must exist under models/)")
    cs.add_argument("--n", type=int, default=3, help="max examples to print/save per composition")

    he = sub.add_parser(
        "hard-examples",
        help="most confidently-wrong predictions on a plain test set, with article text (see module docstring)",
    )
    he.add_argument("--model", choices=["lr", "svm", "cnn", "bert"], default="svm")
    he.add_argument("--comp", nargs="+", default=["real_syn"],
                     help="compositions to check (checkpoints must exist under models/)")
    he.add_argument("--dataset", default="liar", help="'liar', 'welfake', or a raw test-set stem")
    he.add_argument("--n", type=int, default=5, help="max examples to print/save per composition")

    args = ap.parse_args()
    if args.command == "master":
        cmd_master(args)
    elif args.command == "error-analysis":
        cmd_error_analysis(args)
    elif args.command == "cross-target":
        cmd_cross_target(args)
    elif args.command == "seed-summary":
        cmd_seed_summary(args)
    elif args.command == "case-studies":
        cmd_case_studies(args)
    elif args.command == "hard-examples":
        cmd_hard_examples(args)


if __name__ == "__main__":
    main()
