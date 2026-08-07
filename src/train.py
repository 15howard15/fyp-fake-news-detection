"""
train.py -- unified training entry point for LR/SVM, CNN, and BERT.

Merges the old 05_train_traditional.py, 06_train_cnn.py, 07_train_bert.py,
18_train_style_robust.py and 19_train_multisource.py (deleted after this
file was verified to reproduce their output exactly) into one script
controlled by --model / --dataset. Those five scripts were the same
train-one-composition-and-evaluate-on-test_crossdomain loop, copy-pasted per
(model, dataset) combination -- most visibly, 18 and 19 were identical
except for one string constant. --model/--dataset picks the combination
instead of picking a file.

Examples:
    python train.py --model all                          # LR/SVM/CNN/BERT x real_real/mixed/real_syn
    python train.py --model bert                          # BERT only, x real_real/mixed/real_syn
    python train.py --model bert --dataset multisource     # BERT on real_syn_multisource
    python train.py --model cnn --dataset style_robust
    python train.py --model lr_svm --dataset real_real mixed real_syn style_robust real_syn_multisource
    python train.py --model bert --dataset real_real --seed 123 --grad_accum 2

Every --dataset value is a composition name: data/processed/train_<name>.csv
must already exist (build it with build_core_datasets.py / build_style_robust_
dataset.py / build_multisource_dataset.py first). "multisource" is
accepted as a short alias for real_syn_multisource. All datasets are always
evaluated on test_crossdomain.csv, matching every one of the five merged
scripts.
"""
import argparse
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Dataset

import config as cfg
from metrics import compute_metrics, print_metrics, save_metrics
from preprocessing import clean_series

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_ALIASES = {"multisource": "real_syn_multisource"}
DATASET_REQUIRES = {
    "real_syn_multisource": "build_multisource_dataset.py",
    "style_robust": "build_style_robust_dataset.py",
}


def load_test():
    return pd.read_csv(cfg.PROCESSED_DIR / "test_crossdomain.csv")


def load_train(name):
    path = cfg.PROCESSED_DIR / f"train_{name}.csv"
    if not path.exists():
        needs = DATASET_REQUIRES.get(name)
        hint = f" Run {needs} first." if needs else ""
        raise FileNotFoundError(f"{path} not found.{hint}")
    return pd.read_csv(path)


# ----------------------------------------------------------------------
# LR + SVM (TF-IDF)
# ----------------------------------------------------------------------
def train_lr_svm(train, test, comp):
    train_c = train.copy(); train_c["clean"] = clean_series(train_c["text"])
    test_c = test.copy(); test_c["clean"] = clean_series(test_c["text"])

    vec = TfidfVectorizer(ngram_range=cfg.TFIDF_NGRAM_RANGE, max_features=cfg.TFIDF_MAX_FEATURES)
    X = vec.fit_transform(train_c["clean"]); y = train_c["label"].values
    Xt = vec.transform(test_c["clean"]); yt = test_c["label"].values

    # class_weight="balanced" -- a no-op on the balanced 500:500 compositions
    # (real_real/mixed/real_syn/style_robust/real_syn_multisource), but is
    # what recovers LR on imbalanced ones like train_augmented (handled by
    # run_lr_svm_extra_experiments.py, not here) -- left on unconditionally so
    # every composition trained through this function behaves consistently.
    lr = LogisticRegression(solver=cfg.LR_SOLVER, max_iter=1000, class_weight="balanced").fit(X, y)
    lr_pred, lr_prob = lr.predict(Xt), lr.predict_proba(Xt)[:, 1]
    lr_m = compute_metrics(yt, lr_pred, lr_prob)
    save_metrics(lr_m, "LR", comp)
    print_metrics(lr_m, f"\n[LR | {comp}]")

    # method="isotonic", NOT the default "sigmoid" (Platt scaling). Platt
    # scaling fits a 1D logistic regression on the decision scores, which
    # degenerates when those scores are close to perfectly separable --
    # common for high-dim TF-IDF text. Verified on train_mixed: sigmoid
    # calibration inverted the ranking (raw decision_function AUC=0.986,
    # sigmoid-calibrated AUC=0.014, almost exactly 1 - 0.986); isotonic
    # gives AUC=0.966, consistent with the raw decision function.
    svm = CalibratedClassifierCV(LinearSVC(), cv=3, method="isotonic").fit(X, y)
    svm_pred, svm_prob = svm.predict(Xt), svm.predict_proba(Xt)[:, 1]
    svm_m = compute_metrics(yt, svm_pred, svm_prob)
    save_metrics(svm_m, "SVM", comp)
    print_metrics(svm_m, f"\n[SVM | {comp}]")

    joblib.dump(vec, cfg.MODELS_DIR / f"tfidf_{comp}.joblib")
    joblib.dump(lr, cfg.MODELS_DIR / f"lr_{comp}.joblib")
    joblib.dump(svm, cfg.MODELS_DIR / f"svm_{comp}.joblib")


# ----------------------------------------------------------------------
# CNN (GloVe embeddings)
# ----------------------------------------------------------------------
class Vocab:
    def __init__(self, texts, max_size):
        counter = Counter()
        for t in texts:
            counter.update(t.split())
        self.itos = ["<pad>", "<unk>"] + [w for w, _ in counter.most_common(max_size - 2)]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, text, max_len):
        ids = [self.stoi.get(w, 1) for w in text.split()[:max_len]]
        ids += [0] * (max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.itos)


def load_glove(vocab, dim):
    import gensim.downloader as api
    print("Loading GloVe (first time downloads ~130MB)...")
    glove = api.load(f"glove-wiki-gigaword-{dim}")
    matrix = np.random.normal(scale=0.1, size=(len(vocab), dim)).astype("float32")
    matrix[0] = 0.0
    hits = 0
    for word, idx in vocab.stoi.items():
        if word in glove:
            matrix[idx] = glove[word]
            hits += 1
    print(f"  GloVe coverage: {hits}/{len(vocab)} tokens")
    return torch.tensor(matrix)


class CNNDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.x = [vocab.encode(t, max_len) for t in texts]
        self.y = list(labels)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.tensor(self.x[i]), torch.tensor(self.y[i])


class TextCNN(nn.Module):
    def __init__(self, embed_matrix, num_filters, filter_sizes, dropout, n_classes=2):
        super().__init__()
        _, embed_dim = embed_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(embed_matrix, freeze=False, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, fs) for fs in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), n_classes)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        pooled = [torch.relu(c(x)).max(dim=2).values for c in self.convs]
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))


_CNN_VOCAB_CACHE = {}


def get_cnn_vocab_and_embed():
    # Always built from train_real_real, and cached for the life of one
    # `python train.py` run -- GloVe loading is the slow part, and every
    # composition needs the SAME starting vocab/embeddings for CNN results
    # to be comparable (see README "Evaluation setup note").
    #
    # Returns a FRESH .clone() of the embedding matrix on every call, not the
    # cached tensor itself. nn.Embedding.from_pretrained(..., freeze=False)
    # (in TextCNN below) does NOT copy the tensor you hand it -- it shares
    # the same underlying memory (verified: emb.weight.data_ptr() ==
    # base.data_ptr() is True right after construction). Since the CNN then
    # fine-tunes that embedding during training, reusing the cached tensor
    # directly across multiple compositions in one run would let the FIRST
    # composition's fine-tuning leak into every later composition's "fresh"
    # starting point -- real_real would get clean GloVe, mixed would
    # actually start from real_real's already-adjusted vectors, real_syn
    # from mixed's, etc. Call this once per composition/model you train (not
    # once for a whole run) and you always get an independent copy.
    if "vocab" not in _CNN_VOCAB_CACHE:
        base_text = clean_series(pd.read_csv(cfg.PROCESSED_DIR / "train_real_real.csv")["text"])
        vocab = Vocab(base_text, cfg.CNN_VOCAB_SIZE)
        embed = load_glove(vocab, cfg.CNN_EMBED_DIM)
        _CNN_VOCAB_CACHE["vocab"] = vocab
        _CNN_VOCAB_CACHE["embed"] = embed
    return _CNN_VOCAB_CACHE["vocab"], _CNN_VOCAB_CACHE["embed"].clone()


def train_cnn(train, test, comp):
    # Reseed before EACH composition -- see the matching comment in train_bert().
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    vocab, embed = get_cnn_vocab_and_embed()

    train_c = clean_series(train["text"])
    test_c = clean_series(test["text"])
    gen = torch.Generator().manual_seed(cfg.SEED)
    train_dl = DataLoader(CNNDataset(train_c, train["label"], vocab, cfg.CNN_MAX_LEN),
                           batch_size=cfg.CNN_BATCH_SIZE, shuffle=True, generator=gen)
    test_dl = DataLoader(CNNDataset(test_c, test["label"], vocab, cfg.CNN_MAX_LEN),
                          batch_size=cfg.CNN_BATCH_SIZE)

    model = TextCNN(embed, cfg.CNN_NUM_FILTERS, cfg.CNN_FILTER_SIZES, cfg.CNN_DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.CNN_LR)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(cfg.CNN_EPOCHS):
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        print(f"  epoch {epoch+1}/{cfg.CNN_EPOCHS}")

    model.eval()
    probs, preds, ys = [], [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            logits = model(xb.to(DEVICE))
            probs.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy())
            preds.extend(logits.argmax(1).cpu().numpy())
            ys.extend(yb.numpy())
    m = compute_metrics(np.array(ys), np.array(preds), np.array(probs))
    save_metrics(m, "CNN", comp)
    print_metrics(m, f"\n[CNN | {comp}]")
    torch.save(model.state_dict(), cfg.MODELS_DIR / f"cnn_{comp}.pt")


# ----------------------------------------------------------------------
# BERT
# ----------------------------------------------------------------------
class BertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.enc = tokenizer(list(texts), truncation=True, padding="max_length",
                              max_length=max_len, return_tensors="pt")
        self.labels = torch.tensor(list(labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"input_ids": self.enc["input_ids"][i],
                "attention_mask": self.enc["attention_mask"][i],
                "labels": self.labels[i]}


_BERT_TOKENIZER_CACHE = {}


def get_bert_tokenizer():
    if "tok" not in _BERT_TOKENIZER_CACHE:
        from transformers import AutoTokenizer
        _BERT_TOKENIZER_CACHE["tok"] = AutoTokenizer.from_pretrained(cfg.BERT_MODEL_NAME)
    return _BERT_TOKENIZER_CACHE["tok"]


def train_bert(train, test, comp, grad_accum=1, seed=None):
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

    # Reseed before EACH composition -- without this, a composition's
    # starting RNG state depends on how many random draws every composition
    # trained before it in this run consumed, so results would silently
    # shift if --dataset order changed.
    seed = cfg.SEED if seed is None else seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    tokenizer = get_bert_tokenizer()
    train_c = clean_series(train["text"], aggressive=False)   # BERT: minimal cleaning only
    test_c = clean_series(test["text"], aggressive=False)
    gen = torch.Generator().manual_seed(seed)
    train_dl = DataLoader(BertDataset(train_c, train["label"], tokenizer, cfg.BERT_MAX_LEN),
                           batch_size=cfg.BERT_BATCH_SIZE, shuffle=True, generator=gen)
    test_dl = DataLoader(BertDataset(test_c, test["label"], tokenizer, cfg.BERT_MAX_LEN),
                          batch_size=cfg.BERT_BATCH_SIZE)

    model = AutoModelForSequenceClassification.from_pretrained(cfg.BERT_MODEL_NAME, num_labels=2).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.BERT_LR)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    # Linear warmup (10% of steps) + linear decay, plus gradient clipping.
    # Fine-tuning a pretrained transformer with a randomly-initialized
    # classification head at a flat LR from step 1 is a known source of
    # early-training instability -- large gradients through the head can
    # disrupt the pretrained weights before they've had a chance to adapt.
    steps_per_epoch = -(-len(train_dl) // grad_accum)  # ceil div
    total_steps = steps_per_epoch * cfg.BERT_EPOCHS
    warmup_steps = max(1, int(0.1 * total_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    model.train()
    for epoch in range(cfg.BERT_EPOCHS):
        running = 0
        optimizer.zero_grad()
        for step, batch in enumerate(train_dl):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                out = model(**batch)
                loss = out.loss / grad_accum
            scaler.scale(loss).backward()
            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            running += loss.item() * grad_accum
        print(f"  epoch {epoch+1}/{cfg.BERT_EPOCHS}  train_loss={running/len(train_dl):.4f}")

    model.eval()
    probs, preds, ys = [], [], []
    with torch.no_grad():
        for batch in test_dl:
            y = batch.pop("labels")
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                logits = model(**batch).logits
            probs.extend(torch.softmax(logits.float(), 1)[:, 1].cpu().numpy())
            preds.extend(logits.argmax(1).cpu().numpy())
            ys.extend(y.numpy())

    m = compute_metrics(np.array(ys), np.array(preds), np.array(probs))
    # Non-default seeds get their own label so repeat runs don't overwrite
    # the primary result -- lets you report mean +/- std across seeds.
    label = comp if seed == cfg.SEED else f"{comp}_seed{seed}"
    save_metrics(m, "BERT", label)
    print_metrics(m, f"\n[BERT | {label}]")
    model.save_pretrained(cfg.MODELS_DIR / f"bert_{label}")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--model", nargs="+", required=True, choices=["lr_svm", "cnn", "bert", "all"],
        help="which model(s) to train (required -- no default, so a bare "
             "`python train.py` can't accidentally kick off all four models)",
    )
    ap.add_argument(
        "--dataset", nargs="+", default=list(cfg.COMPOSITIONS),
        help=f"composition name(s), i.e. train_<name>.csv under data/processed/. "
             f"Defaults to {list(cfg.COMPOSITIONS)}. 'multisource' is an alias "
             f"for real_syn_multisource.",
    )
    ap.add_argument("--grad_accum", type=int, default=1,
                     help="BERT gradient accumulation steps (raise if OOM)")
    ap.add_argument("--seed", type=int, default=cfg.SEED,
                     help="override cfg.SEED for a BERT repeat run (results saved "
                          "under a separate '<dataset>_seed<N>' label, not overwritten)")
    ap.add_argument("--max-words", type=int, default=None, metavar="N",
                     help="truncate every train AND test document to its first N words "
                          "before any model sees it, and save results under a separate "
                          "'<dataset>_max<N>' label. Use this for a FAIR model-family "
                          "comparison: by default the four models read different amounts "
                          "of each article (TF-IDF has no length limit so LR/SVM read all "
                          "of it, BERT truncates at 512 tokens, CNN at 300), which is an "
                          "uncontrolled confound when comparing architectures.")
    args = ap.parse_args()

    models = ["lr_svm", "cnn", "bert"] if "all" in args.model else args.model
    datasets = [DATASET_ALIASES.get(d, d) for d in args.dataset]

    print(f"Device: {DEVICE}")
    test = load_test()

    def _truncate(df):
        out = df.copy()
        out["text"] = out["text"].astype(str).apply(
            lambda s: " ".join(s.split()[:args.max_words]))
        return out

    if args.max_words:
        print(f"Matched-length mode: every document truncated to {args.max_words} words "
              f"for ALL models (train and test).")
        test = _truncate(test)

    for comp in datasets:
        label = comp if not args.max_words else f"{comp}_max{args.max_words}"
        print(f"\n{'='*50}\nComposition: {label}\n{'='*50}")
        train = load_train(comp)
        if args.max_words:
            train = _truncate(train)
            comp = label  # keeps checkpoints/metrics separate from the full-text runs
        if "lr_svm" in models:
            train_lr_svm(train, test, comp)
        if "cnn" in models:
            train_cnn(train, test, comp)
        if "bert" in models:
            train_bert(train, test, comp, grad_accum=args.grad_accum, seed=args.seed)
    print("\nDone. Metrics in results/, models in models/.")


if __name__ == "__main__":
    main()
