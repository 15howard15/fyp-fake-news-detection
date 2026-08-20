"""evaluate.py -- unified evaluation entry point."""
import argparse
import json
import sys

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
from sklearn.metrics import roc_auc_score

import repro
import config as cfg
from preprocessing import clean_series
from metrics import compute_metrics

EXTRA_DIR = cfg.RESULTS_DIR / "extra"
EXTRA_DIR.mkdir(parents=True, exist_ok=True)

ERROR_ANALYSIS_COMPS = ["real_real", "mixed", "real_syn", "c2_synreal_realfake", "c3_synreal_synfake"]


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
    df = df[df["composition"].isin(cfg.COMPOSITIONS)].reset_index(drop=True)

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
    import torch
    from torch.utils.data import DataLoader
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []

    if "cnn" in models:
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
            repro.set_determinism(cfg.SEED)
            print(f"[CNN] training on {comp} ...")
            tr = pd.read_csv(path)
            tr["clean"] = clean_series(tr["text"])
            gen = torch.Generator().manual_seed(cfg.SEED)
            dl = DataLoader(CNNDataset(tr["clean"], tr["label"], vocab, cfg.CNN_MAX_LEN),
                             batch_size=cfg.CNN_BATCH_SIZE, shuffle=True, generator=gen)
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
        from train import BertDataset

        tok = AutoTokenizer.from_pretrained(cfg.BERT_MODEL_NAME)
        test_clean = clean_series(test_df["text"], aggressive=False)
        tdl = DataLoader(BertDataset(test_clean, test_df["label"], tok, cfg.BERT_MAX_LEN),
                          batch_size=cfg.BERT_BATCH_SIZE)

        for comp in ERROR_ANALYSIS_COMPS:
            path = cfg.PROCESSED_DIR / f"train_{comp}.csv"
            if not path.exists():
                continue
            repro.set_determinism(cfg.SEED)
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


CROSS_TARGET_ALIASES = {
    "welfake": "test_crossdomain2",
    "welfake_clean": "test_crossdomain2_clean",
    "liar": "test_crossdomain",
}


def cmd_cross_target(args):
    test_name = CROSS_TARGET_ALIASES.get(args.dataset, args.dataset)
    test_path = cfg.PROCESSED_DIR / f"{test_name}.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"{test_path} not found. Run build_test_sets.py first.")
    test = pd.read_csv(test_path)
    y = test["label"].values

    rows = []
    for comp in args.comp:
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
    out = EXTRA_DIR / ("crossdomain2_results.csv" if args.dataset == "welfake"
                       else f"crosstarget_{args.dataset}_results.csv")
    if out.exists():
        df = (pd.concat([pd.read_csv(out), df], ignore_index=True)
                .drop_duplicates(subset=["dataset", "comp", "model"], keep="last"))
    df.to_csv(out, index=False)
    print(f"\n=== CROSS-TARGET RESULTS ({args.dataset}) ===")
    print(df.to_string(index=False))
    print(f"\nSaved to {out}")


def predict_labels(comp, model, test):
    """Predicted labels for one (composition, model) pair on a test frame."""
    y = test["label"].values
    if model in ("LR", "SVM"):
        vec = joblib.load(cfg.MODELS_DIR / f"tfidf_{comp}.joblib")
        clf = joblib.load(cfg.MODELS_DIR / f"{'lr' if model == 'LR' else 'svm'}_{comp}.joblib")
        return clf.predict(vec.transform(clean_series(test["text"])))

    if model == "CNN":
        import torch
        from torch.utils.data import DataLoader
        from train import CNNDataset, TextCNN, get_cnn_vocab_and_embed
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        vocab, embed = get_cnn_vocab_and_embed()
        net = TextCNN(embed, cfg.CNN_NUM_FILTERS, cfg.CNN_FILTER_SIZES, cfg.CNN_DROPOUT).to(DEVICE)
        net.load_state_dict(torch.load(cfg.MODELS_DIR / f"cnn_{comp}.pt", map_location=DEVICE))
        net.eval()
        dl = DataLoader(CNNDataset(clean_series(test["text"]), y, vocab, cfg.CNN_MAX_LEN),
                        batch_size=cfg.CNN_BATCH_SIZE)
        preds = []
        with torch.no_grad():
            for xb, _ in dl:
                preds.extend(net(xb.to(DEVICE)).argmax(1).cpu().numpy())
        return np.array(preds)

    if model == "BERT":
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from train import BertDataset
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(cfg.BERT_MODEL_NAME)
        net = AutoModelForSequenceClassification.from_pretrained(
            cfg.MODELS_DIR / f"bert_{comp}").to(DEVICE)
        net.eval()
        dl = DataLoader(BertDataset(clean_series(test["text"], aggressive=False), y,
                                    tok, cfg.BERT_MAX_LEN), batch_size=cfg.BERT_BATCH_SIZE)
        preds = []
        with torch.no_grad():
            for batch in dl:
                batch.pop("labels")
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                    preds.extend(net(**batch).logits.argmax(1).cpu().numpy())
        return np.array(preds)

    raise ValueError(f"unknown model {model!r}")


def cmd_edit_distance(args):
    """How heavily was each synthetic corpus rewritten from its source?"""
    import difflib

    rows = []
    for name in args.files:
        p = cfg.SYNTHETIC_DIR / (name if name.endswith(".csv") else f"{name}.csv")
        if not p.exists():
            print(f"  (skip {p.name} -- not found)")
            continue
        df = pd.read_csv(p)
        if not {"text", "source_text"} <= set(df.columns):
            print(f"  (skip {p.name} -- needs both text and source_text)")
            continue
        if args.n:
            df = df.head(args.n)
        sims = []
        for src, gen in zip(df["source_text"], df["text"]):
            a = str(src)[:args.window]
            sims.append(difflib.SequenceMatcher(None, a, str(gen),
                                                autojunk=False).ratio())
        s = np.array(sims)
        rows.append({"file": p.stem, "n": len(s),
                     "similarity_mean": round(float(s.mean()), 4),
                     "similarity_sd": round(float(s.std(ddof=1)), 4),
                     "similarity_median": round(float(np.median(s)), 4),
                     "similarity_p25": round(float(np.percentile(s, 25)), 4),
                     "similarity_p75": round(float(np.percentile(s, 75)), 4),
                     "edit_distance_mean": round(float(1 - s.mean()), 4)})

    if not rows:
        print("No corpora measured.")
        return

    df = pd.DataFrame(rows)
    print("\n=== EDIT DISTANCE FROM SOURCE "
          f"(first {args.window:,} chars, the window the prompt saw) ===")
    print(df.to_string(index=False))

    print("\n=== PAIRWISE ASYMMETRY (percentage points of similarity) ===")
    pair_rows = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            a, b = df.iloc[i], df.iloc[j]
            gap = abs(a.similarity_mean - b.similarity_mean) * 100
            verdict = "OK" if gap <= args.tolerance else "ASYMMETRIC"
            print(f"  {a.file:28s} vs {b.file:28s} "
                  f"{a.similarity_mean:.3f} vs {b.similarity_mean:.3f}  "
                  f"gap={gap:5.1f}pp  [{verdict}]")
            pair_rows.append({"file": f"{a.file} vs {b.file}", "n": "",
                              "similarity_mean": "", "similarity_sd": "",
                              "similarity_median": "", "similarity_p25": "",
                              "similarity_p75": "",
                              "edit_distance_mean": "",
                              "gap_pp": round(gap, 2),
                              "within_tolerance": bool(gap <= args.tolerance)})

    out = EXTRA_DIR / "edit_distance.csv"
    pd.concat([df, pd.DataFrame(pair_rows)], ignore_index=True).to_csv(out, index=False)
    print(f"\nTolerance: {args.tolerance} percentage points. Saved to {out}")


def cmd_significance(args):
    """McNemar's test on pairs of already-trained models."""
    from statsmodels.stats.contingency_tables import mcnemar

    test_name = CROSS_TARGET_ALIASES.get(args.dataset, args.dataset)
    test_path = cfg.PROCESSED_DIR / f"{test_name}.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"{test_path} not found. Run build_test_sets.py first.")
    test = pd.read_csv(test_path).reset_index(drop=True)
    y = test["label"].values
    print(f"Test set: {test_name} ({len(test):,} rows)")

    correct, missing = {}, []
    for comp in args.comp:
        for model in args.model:
            try:
                pred = predict_labels(comp, model, test)
            except (FileNotFoundError, OSError) as e:
                missing.append(f"{model}/{comp}")
                print(f"  (skip {model:4s} / {comp} -- no checkpoint)")
                continue
            correct[(model, comp)] = (np.asarray(pred) == y).astype(int)
            print(f"  loaded {model:4s} / {comp:22s} accuracy={correct[(model, comp)].mean():.4f}")

    def test_pair(a_key, b_key, label_a, label_b, model_col, axis, held):
        ca, cb = correct[a_key], correct[b_key]
        b = int(((ca == 1) & (cb == 0)).sum())
        c = int(((ca == 0) & (cb == 1)).sum())
        n_disc = b + c
        if n_disc == 0:
            return {"comparison_a": label_a, "comparison_b": label_b, "model": model_col,
                    "axis": axis, "held_constant": held, "dataset": args.dataset,
                    "n_samples": len(y), "n_discordant": 0, "b_a_right_b_wrong": 0,
                    "c_a_wrong_b_right": 0, "accuracy_a": round(float(ca.mean()), 4),
                    "accuracy_b": round(float(cb.mean()), 4), "statistic": None,
                    "p_value": 1.0, "significant_at_0.05": False, "exact": True}
        res = mcnemar([[int(((ca == 1) & (cb == 1)).sum()), b],
                       [c, int(((ca == 0) & (cb == 0)).sum())]],
                      exact=(n_disc < args.exact_below),
                      correction=(n_disc >= args.exact_below))
        return {"comparison_a": label_a, "comparison_b": label_b, "model": model_col,
                "axis": axis, "held_constant": held, "dataset": args.dataset,
                "n_samples": len(y), "n_discordant": n_disc,
                "b_a_right_b_wrong": b, "c_a_wrong_b_right": c,
                "accuracy_a": round(float(ca.mean()), 4),
                "accuracy_b": round(float(cb.mean()), 4),
                "statistic": round(float(res.statistic), 4),
                "p_value": float(res.pvalue),
                "significant_at_0.05": bool(res.pvalue < 0.05),
                "exact": bool(n_disc < args.exact_below)}

    rows = []
    if args.axis in ("recipe", "both"):
        for model in args.model:
            comps = [c for c in args.comp if (model, c) in correct]
            for i in range(len(comps)):
                for j in range(i + 1, len(comps)):
                    rows.append(test_pair((model, comps[i]), (model, comps[j]),
                                          comps[i], comps[j], model, "recipe", model))
    if args.axis in ("model", "both"):
        for comp in args.comp:
            models = [m for m in args.model if (m, comp) in correct]
            for i in range(len(models)):
                for j in range(i + 1, len(models)):
                    rows.append(test_pair((models[i], comp), (models[j], comp),
                                          models[i], models[j],
                                          f"{models[i]} vs {models[j]}", "model", comp))

    if not rows:
        print("\nNo comparisons could be made -- no checkpoints found.")
        return

    df = pd.DataFrame(rows)

    order = df["p_value"].values.argsort()
    n = len(df)
    adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * df["p_value"].values[idx])
        adj[idx] = min(1.0, running)
    df["p_value_holm"] = adj
    df["significant_holm"] = df["p_value_holm"] < 0.05

    out = cfg.RESULTS_DIR / "statistical_significance.csv"
    if out.exists():
        prev = pd.read_csv(out)
        df = (pd.concat([prev, df], ignore_index=True)
                .drop_duplicates(subset=["dataset", "axis", "comparison_a",
                                         "comparison_b", "model", "held_constant"],
                                 keep="last"))
    df.to_csv(out, index=False)

    print(f"\n=== McNEMAR'S TEST ({args.dataset}, {len(df)} comparisons) ===")
    show = df[["axis", "comparison_a", "comparison_b", "model", "n_discordant",
               "accuracy_a", "accuracy_b", "p_value", "significant_at_0.05",
               "significant_holm"]].copy()
    show["p_value"] = show["p_value"].map(lambda v: f"{v:.3g}")
    print(show.to_string(index=False))
    n_raw = int(df["significant_at_0.05"].sum())
    n_holm = int(df["significant_holm"].sum())
    print(f"\n{n_raw}/{len(df)} significant at raw p<0.05; "
          f"{n_holm}/{len(df)} survive Holm-Bonferroni correction.")
    if missing:
        print(f"Skipped (no checkpoint): {', '.join(missing)}")
    print(f"Saved to {out}")


def cmd_hard_examples(args):
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


def cmd_length_sweep(args):
    """Separate the LENGTH confound from the DOMAIN confound in RQ3."""
    test_path = cfg.PROCESSED_DIR / "test_crossdomain2.csv"
    if not test_path.exists():
        print(f"{test_path} not found -- run build_test_sets.py with WELFake available.")
        return
    base = pd.read_csv(test_path)

    liar = pd.read_csv(cfg.PROCESSED_DIR / "test_crossdomain.csv")
    liar_fake_len = liar[liar.label == 1]["text"].astype(str).str.split().str.len().median()
    print(f"Median LIAR fake-statement length: {liar_fake_len:.0f} words")
    print("Truncating the WELFake test set to each length below and re-scoring "
          "the same checkpoints (domain held constant, only length varies).\n")

    lengths = args.lengths or [None, 300, 150, 75, 40, 20]
    rows = []
    for comp in args.comp:
        for model in args.model:
            fname = "lr" if model == "lr" else "svm"
            vec_p = cfg.MODELS_DIR / f"tfidf_{comp}.joblib"
            clf_p = cfg.MODELS_DIR / f"{fname}_{comp}.joblib"
            if not (vec_p.exists() and clf_p.exists()):
                print(f"  (skipping {model}/{comp} -- checkpoint not found)")
                continue
            vec, clf = joblib.load(vec_p), joblib.load(clf_p)
            print(f"=== {model.upper()} | {comp} ===")
            for L in lengths:
                if L is None:
                    txt = base["text"].astype(str)
                    tag = "full"
                else:
                    def _cut(s):
                        return " ".join(str(s).split()[:L])
                    if args.truncate_class == "fake":
                        txt = base.apply(
                            lambda r: _cut(r["text"]) if r["label"] == 1
                            else str(r["text"]), axis=1)
                        tag = f"{L}w/fake"
                    else:
                        txt = base["text"].astype(str).apply(_cut)
                        tag = f"{L}w"
                X = vec.transform(clean_series(txt))
                m = compute_metrics(base["label"].values, clf.predict(X),
                                    clf.predict_proba(X)[:, 1])
                print(f"  truncated to {tag:>5s}: F1={m['f1']:.3f}  AUC={m['auc_roc']:.3f}")
                rows.append({"model": model.upper(), "comp": comp,
                             "truncated_to": tag, "f1": round(m["f1"], 4),
                             "auc_roc": round(m["auc_roc"], 4)})
            print()

    out = EXTRA_DIR / "length_sweep_results.csv"
    df_new = pd.DataFrame(rows)
    if out.exists():
        df_new = pd.concat([pd.read_csv(out), df_new], ignore_index=True) \
                   .drop_duplicates(subset=["model", "comp", "truncated_to"], keep="last")
    df_new.to_csv(out, index=False)
    print(f"Saved to {out} ({len(df_new)} rows)")


def cmd_leakage(args):
    tests = {}
    for stem in ("test_indomain", "test_crossdomain", "test_crossdomain2",
                 "test_crossdomain2_clean"):
        p = cfg.PROCESSED_DIR / f"{stem}.csv"
        if p.exists():
            tests[stem] = pd.read_csv(p)
    if not tests:
        print("No test sets found. Run build_test_sets.py first.")
        return

    train_paths = sorted(cfg.PROCESSED_DIR.glob("train_*.csv"))
    if not train_paths:
        print("No train_*.csv found. Run the build_*_datasets.py scripts first.")
        return

    FULL_POOL_COMPS = {
        "train_augmented_full",
        "train_c6_full_augmented",
        "train_lowres_real",
        "train_lowres_aug",
    }

    test_texts = {k: set(v["text"].astype(str)) for k, v in tests.items()}

    rows = []
    worst_pct = 0.0
    worst_name = "-"
    print("\n=== 1. TRAIN/TEST CONTAMINATION (exact text matches) ===")
    print("  (* = draws on the full ISOT pool by design; excluded from the threshold)")
    header = f"{'train file':34s}" + "".join(f"{k.replace('test_', ''):>16s}" for k in tests)
    print(header)
    for tp in train_paths:
        tr = set(pd.read_csv(tp)["text"].astype(str))
        by_design = tp.stem in FULL_POOL_COMPS
        cells = []
        for stem, tt in test_texts.items():
            n = len(tr & tt)
            pct = 100.0 * n / max(len(tt), 1)
            if not by_design and pct > worst_pct:
                worst_pct, worst_name = pct, f"{tp.stem} vs {stem}"
            cells.append(f"{n} ({pct:.2f}%)")
            rows.append({"check": "train_test_overlap", "train": tp.stem,
                         "test": stem, "n_overlap": n, "pct_of_test": round(pct, 3),
                         "by_design_full_pool": by_design})
        label = f"{tp.stem}{' *' if by_design else ''}"
        print(f"{label:34s}" + "".join(f"{c:>16s}" for c in cells))

    print("\n=== 2. CORPUS INDEPENDENCE (is the test corpus really unseen?) ===")
    isot_all = set()
    for stem in ("isot_real", "isot_fake"):
        p = cfg.PROCESSED_DIR / f"{stem}.csv"
        if p.exists():
            isot_all |= set(pd.read_csv(p)["text"].astype(str))

    if isot_all:
        for stem, df in tests.items():
            if "source" not in df.columns:
                continue
            for src, g in df.groupby("source"):
                if src == "isot":
                    continue
                n = g["text"].astype(str).isin(isot_all).sum()
                pct = 100.0 * n / max(len(g), 1)
                print(f"  {stem:22s} source={src:10s} {n:5d}/{len(g):5d} "
                      f"({pct:5.1f}%) of its rows also exist in ISOT")
                rows.append({"check": "corpus_overlap_with_isot", "train": "-",
                             "test": f"{stem}:{src}", "n_overlap": int(n),
                             "pct_of_test": round(pct, 3)})

    print("\n=== 3. DUPLICATE ARTICLES WITHIN EACH SOURCE CORPUS ===")
    for stem in ("isot_real", "isot_fake", "liar_fake", "welfake_fake"):
        p = cfg.PROCESSED_DIR / f"{stem}.csv"
        if not p.exists():
            continue
        t = pd.read_csv(p)["text"].astype(str)
        dup = int(t.duplicated().sum())
        pct = 100.0 * dup / max(len(t), 1)
        print(f"  {stem:16s} {len(t):7,} rows, {t.nunique():7,} unique, "
              f"{dup:6,} duplicated ({pct:.1f}%)")
        rows.append({"check": "within_corpus_duplicates", "train": "-",
                     "test": stem, "n_overlap": dup, "pct_of_test": round(pct, 3)})

    print("\n=== 4. LENGTH SHORTCUT IN EACH TEST SET ===")
    print("  How well does a 'classifier' that knows ONLY the document's word")
    print("  count separate the classes? 0.5 = length is uninformative.")
    print("  A high number means a good score on that test set is not by itself")
    print("  evidence of anything, because word-counting would also score well.")
    for stem, df in tests.items():
        if "label" not in df.columns or df["label"].nunique() < 2:
            continue
        w = df["text"].astype(str).str.split().str.len()
        auc = roc_auc_score(df["label"], -w)
        med_r = int(w[df.label == cfg.LABEL_REAL].median())
        med_f = int(w[df.label == cfg.LABEL_FAKE].median())
        flag = "  <-- length alone nearly solves this set" if auc > 0.9 or auc < 0.1 else ""
        print(f"  {stem:24s} AUC={auc:.4f}   median words: "
              f"real {med_r:4d} / fake {med_f:4d}{flag}")
        rows.append({"check": "length_shortcut", "train": "-", "test": stem,
                     "n_overlap": "", "pct_of_test": round(float(auc), 4)})

    out = EXTRA_DIR / "leakage_report.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved full report to {out}")

    if worst_pct > args.max_pct:
        raise SystemExit(
            f"\nFAIL: train/test overlap reached {worst_pct:.2f}% of a test set "
            f"({worst_name}), above the --max-pct {args.max_pct}% threshold.\n"
            f"Expected baseline is well under 2% and comes from ISOT's own duplicate "
            f"articles. A figure above the threshold means the split logic changed -- "
            f"check build_test_sets.py and the build_*_datasets.py scripts."
        )
    print(f"PASS: worst train/test overlap among threshold-checked compositions is "
          f"{worst_pct:.2f}% ({worst_name}), within the {args.max_pct}% threshold "
          f"and consistent with ISOT's known duplicate articles.")


def cmd_seed_summary(args):
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
                     help="'welfake' (-> test_crossdomain2.csv), 'welfake_clean' "
                          "(ISOT-overlapping articles removed), 'liar', or a raw test-set stem")
    ct.add_argument("--comp", nargs="+", default=list(cfg.COMPOSITIONS),
                     help="composition name(s) to evaluate (checkpoints must exist under models/)")

    ed = sub.add_parser(
        "edit-distance",
        help="how heavily each synthetic corpus was rewritten from its source, "
             "and whether the two classes match (see function docstring)",
    )
    ed.add_argument("--files", nargs="+",
                     default=["synthetic_real", "synthetic_fake"],
                     help="corpus stems under data/synthetic/ (need text + source_text)")
    ed.add_argument("--n", type=int, default=None,
                     help="measure only the first N rows of each (default: all)")
    ed.add_argument("--window", type=int, default=4000,
                     help="compare against the first N characters of the source -- "
                          "must match gen_common.truncate_article's cap (4000), "
                          "which is all the prompt ever saw")
    ed.add_argument("--tolerance", type=float, default=5.0,
                     help="how many percentage points of mean similarity two "
                          "corpora may differ by and still count as symmetric")

    sg = sub.add_parser(
        "significance",
        help="McNemar's test between already-trained checkpoints -- is a score "
             "gap larger than chance? (see function docstring)",
    )
    sg.add_argument("--dataset", default="liar",
                     help="'liar', 'welfake', 'welfake_clean', or a raw test-set stem")
    sg.add_argument("--comp", nargs="+",
                     default=["real_real", "mixed", "real_syn", "style_robust"],
                     help="composition names to compare (checkpoints must exist)")
    sg.add_argument("--model", nargs="+", choices=["LR", "SVM", "CNN", "BERT"],
                     default=["LR", "SVM", "CNN", "BERT"],
                     help="models to include")
    sg.add_argument("--axis", choices=["recipe", "model", "both"], default="both",
                     help="'recipe' = same model, two training recipes; "
                          "'model' = same recipe, two models; 'both' = all of it")
    sg.add_argument("--exact-below", type=int, default=25,
                     help="use the exact binomial test when the discordant count "
                          "is under this (default 25); above it, chi-square with "
                          "continuity correction")

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

    ls = sub.add_parser(
        "length-sweep",
        help="truncate the WELFake test set to varying lengths to separate the length "
             "confound from the domain confound in RQ3 (see function docstring)",
    )
    ls.add_argument("--model", nargs="+", choices=["lr", "svm"], default=["lr", "svm"],
                     help="LR/SVM only -- deterministic and inference-only, so the "
                          "comparison isn't muddied by CNN/BERT run-to-run variance")
    ls.add_argument("--comp", nargs="+", default=["real_real", "mixed"],
                     help="composition checkpoints to sweep")
    ls.add_argument("--lengths", nargs="+", type=int, default=None,
                     help="word limits to test (default: full 300 150 75 40 20)")
    ls.add_argument("--truncate-class", choices=["both", "fake"], default="both",
                     help="'both' shortens the whole test set; 'fake' shortens only the "
                          "fake class, reproducing LIAR's long-real/short-fake asymmetry")

    lk = sub.add_parser(
        "leakage",
        help="verify no train text appears in any test set + measure ISOT overlap of each test corpus",
    )
    lk.add_argument("--max-pct", type=float, default=2.0,
                     help="fail if train/test overlap exceeds this %% of a test set "
                          "(default 2.0 -- ISOT's own duplicates put the baseline near 1%%)")

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
    elif args.command == "leakage":
        cmd_leakage(args)
    elif args.command == "length-sweep":
        cmd_length_sweep(args)
    elif args.command == "significance":
        cmd_significance(args)
    elif args.command == "edit-distance":
        cmd_edit_distance(args)


if __name__ == "__main__":
    main()
