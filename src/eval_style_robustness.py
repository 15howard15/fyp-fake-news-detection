
"""
eval_style_robustness.py -- Objective 4 evaluation: for each already-
trained model/composition, compare performance on the ORIGINAL held-out
articles vs. the STYLE-ATTACKED versions of the SAME articles (see
generate_style_attack.py). A robust model's accuracy should barely move;
a model relying on tone/sentiment shortcuts will degrade sharply.

Also reports FLIP RATE: of the articles the model got right originally,
what fraction does it get WRONG after the style attack. This is a more
direct robustness signal than aggregate F1, since it's paired per-article.

No retraining -- reuses the already-saved LR/SVM/CNN/BERT checkpoints for
real_real / mixed / real_syn, so this is cheap to run.
"""
import argparse

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config as cfg
from metrics import compute_metrics
from preprocessing import clean_series
from train import Vocab, CNNDataset, BertDataset

EXTRA_DIR = cfg.RESULTS_DIR / "extra"
EXTRA_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPS = ["real_real", "mixed", "real_syn", "style_robust"]


def load_pair(pair="forward"):
    """Load the paired (original, attacked) evaluation sets, aligned by orig_id.
    pair="forward" (default): real->sensationalized, fake->neutralized -- the
    original Q4 attack (generate_style_attack.py), the direction that
    matches the "fake news sounds dramatic" shortcut a model might be using.
    pair="reverse": real->neutralized, fake->sensationalized (generate_
    style_attack_reverse.py) -- the OPPOSITE direction, to check whether a
    fix like style_robust generalizes to any tone shift or only protects
    against the direction it was built against."""
    if pair == "reverse":
        orig_path = cfg.SYNTHETIC_DIR / "style_attack_reverse_originals.csv"
        attacked_path = cfg.SYNTHETIC_DIR / "style_attack_reverse.csv"
    else:
        orig_path = cfg.SYNTHETIC_DIR / "style_attack_originals.csv"
        attacked_path = cfg.SYNTHETIC_DIR / "style_attack.csv"
    orig = pd.read_csv(orig_path)
    attacked = pd.read_csv(attacked_path)
    orig = orig.sort_values("orig_id").reset_index(drop=True)
    attacked = attacked.sort_values("orig_id").reset_index(drop=True)
    assert (orig["orig_id"].values == attacked["orig_id"].values).all(), "misaligned pairs"
    return orig, attacked


def eval_traditional(orig, attacked, rows):
    o_clean = clean_series(orig["text"])
    a_clean = clean_series(attacked["text"])
    y = orig["label"].values  # same labels in both, by construction
    for comp in COMPS:
        vec = joblib.load(cfg.MODELS_DIR / f"tfidf_{comp}.joblib")
        Xo = vec.transform(o_clean)
        Xa = vec.transform(a_clean)
        for mname, fname in (("LR", "lr"), ("SVM", "svm")):
            model = joblib.load(cfg.MODELS_DIR / f"{fname}_{comp}.joblib")
            po, pa = model.predict(Xo), model.predict(Xa)
            correct_before = po == y
            correct_after = pa == y
            flip_rate = (correct_before & ~correct_after).sum() / max(correct_before.sum(), 1)
            mo = compute_metrics(y, po, model.predict_proba(Xo)[:, 1])
            ma = compute_metrics(y, pa, model.predict_proba(Xa)[:, 1])
            rows.append({"model": mname, "comp": comp,
                         "f1_original": round(mo["f1"], 4), "f1_attacked": round(ma["f1"], 4),
                         "acc_original": round(mo["accuracy"], 4), "acc_attacked": round(ma["accuracy"], 4),
                         "flip_rate": round(flip_rate, 4)})
            print(f"  [{mname:4s}|{comp:10s}] F1 {mo['f1']:.3f}->{ma['f1']:.3f}  flip_rate={flip_rate:.3f}")


# Vocab/CNNDataset shared with train.py (identical code). TextCNN below stays
# LOCAL and separate from train.py's -- this one builds an empty, randomly-
# initialized embedding shell sized (vocab_size, embed_dim) whose weights get
# fully overwritten by load_state_dict() right after construction (this
# script only evaluates already-trained checkpoints, it never trains). train.
# py's TextCNN instead requires an actual pretrained embedding matrix up
# front (nn.Embedding.from_pretrained(embed_matrix, ...)) for training from
# scratch. Different constructor signature, different purpose -- not
# duplication to remove, just two TextCNNs for two different jobs.
class TextCNN(nn.Module):
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


def eval_cnn(orig, attacked, rows):
    # Vocab is deterministic given the same seed + source text, matching how
    # train.py's get_cnn_vocab_and_embed() built it (from train_real_real,
    # shared across comps).
    base_txt = clean_series(pd.read_csv(cfg.PROCESSED_DIR / "train_real_real.csv")["text"])
    vocab = Vocab(base_txt, cfg.CNN_VOCAB_SIZE)
    o_clean = clean_series(orig["text"])
    a_clean = clean_series(attacked["text"])
    y = orig["label"].values
    o_dl = DataLoader(CNNDataset(o_clean, y, vocab, cfg.CNN_MAX_LEN), batch_size=cfg.CNN_BATCH_SIZE)
    a_dl = DataLoader(CNNDataset(a_clean, y, vocab, cfg.CNN_MAX_LEN), batch_size=cfg.CNN_BATCH_SIZE)

    for comp in COMPS:
        model = TextCNN(cfg.CNN_EMBED_DIM, cfg.CNN_NUM_FILTERS, cfg.CNN_FILTER_SIZES,
                        cfg.CNN_DROPOUT, len(vocab)).to(DEVICE)
        model.load_state_dict(torch.load(cfg.MODELS_DIR / f"cnn_{comp}.pt", map_location=DEVICE))
        model.eval()

        def predict(dl):
            probs, preds = [], []
            with torch.no_grad():
                for xb, yb in dl:
                    logits = model(xb.to(DEVICE))
                    probs.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy())
                    preds.extend(logits.argmax(1).cpu().numpy())
            return np.array(preds), np.array(probs)

        po, probo = predict(o_dl)
        pa, proba = predict(a_dl)
        correct_before = po == y
        correct_after = pa == y
        flip_rate = (correct_before & ~correct_after).sum() / max(correct_before.sum(), 1)
        mo = compute_metrics(y, po, probo)
        ma = compute_metrics(y, pa, proba)
        rows.append({"model": "CNN", "comp": comp,
                     "f1_original": round(mo["f1"], 4), "f1_attacked": round(ma["f1"], 4),
                     "acc_original": round(mo["accuracy"], 4), "acc_attacked": round(ma["accuracy"], 4),
                     "flip_rate": round(flip_rate, 4)})
        print(f"  [CNN |{comp:10s}] F1 {mo['f1']:.3f}->{ma['f1']:.3f}  flip_rate={flip_rate:.3f}")


def eval_bert(orig, attacked, rows):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.BERT_MODEL_NAME)
    o_clean = clean_series(orig["text"], aggressive=False)
    a_clean = clean_series(attacked["text"], aggressive=False)
    y = orig["label"].values
    o_dl = DataLoader(BertDataset(o_clean, y, tok, cfg.BERT_MAX_LEN), batch_size=cfg.BERT_BATCH_SIZE)
    a_dl = DataLoader(BertDataset(a_clean, y, tok, cfg.BERT_MAX_LEN), batch_size=cfg.BERT_BATCH_SIZE)

    for comp in COMPS:
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.MODELS_DIR / f"bert_{comp}").to(DEVICE)
        model.eval()

        def predict(dl):
            probs, preds = [], []
            with torch.no_grad():
                for batch in dl:
                    batch.pop("labels")
                    batch = {k: v.to(DEVICE) for k, v in batch.items()}
                    logits = model(**batch).logits
                    probs.extend(torch.softmax(logits.float(), 1)[:, 1].cpu().numpy())
                    preds.extend(logits.argmax(1).cpu().numpy())
            return np.array(preds), np.array(probs)

        po, probo = predict(o_dl)
        pa, proba = predict(a_dl)
        correct_before = po == y
        correct_after = pa == y
        flip_rate = (correct_before & ~correct_after).sum() / max(correct_before.sum(), 1)
        mo = compute_metrics(y, po, probo)
        ma = compute_metrics(y, pa, proba)
        rows.append({"model": "BERT", "comp": comp,
                     "f1_original": round(mo["f1"], 4), "f1_attacked": round(ma["f1"], 4),
                     "acc_original": round(mo["accuracy"], 4), "acc_attacked": round(ma["accuracy"], 4),
                     "flip_rate": round(flip_rate, 4)})
        print(f"  [BERT|{comp:10s}] F1 {mo['f1']:.3f}->{ma['f1']:.3f}  flip_rate={flip_rate:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", choices=["lr_svm", "cnn", "bert"],
                     default=["lr_svm", "cnn", "bert"])
    ap.add_argument("--pair", choices=["forward", "reverse"], default="forward",
                     help="'forward' = real->sensationalized/fake->neutralized (original); "
                          "'reverse' = real->neutralized/fake->sensationalized (generate_style_attack_reverse.py)")
    args = ap.parse_args()

    orig, attacked = load_pair(args.pair)
    types = orig["attack_type"].value_counts().to_dict()
    print(f"Loaded {len(orig)} paired (original, style-attacked) articles [{args.pair}]: {types}")

    rows = []
    if "lr_svm" in args.models:
        eval_traditional(orig, attacked, rows)
    if "cnn" in args.models:
        eval_cnn(orig, attacked, rows)
    if "bert" in args.models:
        eval_bert(orig, attacked, rows)

    df = pd.DataFrame(rows)
    out_name = "style_robustness_results.csv" if args.pair == "forward" else "style_robustness_reverse_results.csv"
    out = EXTRA_DIR / out_name
    df.to_csv(out, index=False)
    print(f"\n=== STYLE ROBUSTNESS RESULTS [{args.pair}] ===")
    print(df.to_string(index=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
