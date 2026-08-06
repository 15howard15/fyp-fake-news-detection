
import argparse
import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

import config as cfg
from preprocessing import clean_series
from metrics import compute_metrics

EXTRA_DIR = cfg.RESULTS_DIR / "extra"
EXTRA_DIR.mkdir(parents=True, exist_ok=True)


def load_test(name):
    df = pd.read_csv(cfg.PROCESSED_DIR / f"{name}.csv")
    df["clean"] = clean_series(df["text"])
    return df


def train_eval(train_name, tests):
    """Train LR + SVM on train_name, eval on each test set. Returns list of rows."""
    train = pd.read_csv(cfg.PROCESSED_DIR / f"{train_name}.csv")
    train["clean"] = clean_series(train["text"])

    vec = TfidfVectorizer(ngram_range=cfg.TFIDF_NGRAM_RANGE,
                          max_features=cfg.TFIDF_MAX_FEATURES)
    X_train = vec.fit_transform(train["clean"])
    y_train = train["label"].values

    # class_weight="balanced" -- the "augmentation" compositions (train_augmented,
    # train_lowres_aug, train_c6_full_augmented) grow the fake class without
    # growing the real class (e.g. 500 real vs 1,000 fake). Verified via direct
    # diagnostic: LR was defaulting to near-"always predict fake" on these
    # (recall~1.0, precision~base-rate) -- class_weight="balanced" reweights the
    # loss to counteract this and recovers real signal (train_augmented crossdomain
    # F1 0.7385->0.9336). It's a no-op on the already-balanced 500:500 compositions
    # (real_real/mixed/real_syn/C2/C3), confirmed identical F1 with/without it.
    lr = LogisticRegression(solver=cfg.LR_SOLVER, max_iter=1000, class_weight="balanced").fit(X_train, y_train)
    # method="isotonic" -- see the comment in train.py's train_lr_svm(). Default
    # "sigmoid" (Platt scaling) can invert the probability ranking when
    # LinearSVC's decision scores are close to perfectly separable.
    # NOTE: class_weight="balanced" was also tested on SVM here (verified via
    # direct diagnostic) and barely moved F1 on the imbalanced augmentation
    # compositions (e.g. train_augmented: 0.7253->0.7253), unlike LR's dramatic
    # recovery -- SVM's failure mode is margin degeneracy on near-duplicate
    # classes (see real_syn), not loss-level class weighting, so it's
    # deliberately NOT added here.
    svm = CalibratedClassifierCV(LinearSVC(), cv=3, method="isotonic").fit(X_train, y_train)

    rows = []
    for test_label, test_df in tests.items():
        Xt = vec.transform(test_df["clean"])
        yt = test_df["label"].values
        for mname, model in (("LR", lr), ("SVM", svm)):
            prob = model.predict_proba(Xt)[:, 1]
            pred = model.predict(Xt)
            m = compute_metrics(yt, pred, prob)
            rows.append({
                "train_set": train_name, "model": mname, "test": test_label,
                "accuracy": round(m["accuracy"], 4),
                "precision": round(m["precision"], 4),
                "recall": round(m["recall"], 4),
                "f1": round(m["f1"], 4),
                "auc_roc": round(m["auc_roc"], 4),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="run Option 4 sweep sets")
    ap.add_argument("--plot", action="store_true", help="save sweep plot")
    args = ap.parse_args()

    tests = {}
    for t in ("test_indomain", "test_crossdomain"):
        path = cfg.PROCESSED_DIR / f"{t}.csv"
        if path.exists():
            tests[t.replace("test_", "")] = load_test(t)
    if not tests:
        # fall back to the original shared test if 04c wasn't run
        tests["crossdomain"] = load_test("test_shared")
        print("Using test_shared as crossdomain (run 04c for in-domain too).")

    # Which training sets to run
    if args.sweep:
        train_sets = [f"train_sweep_{a:04d}" for a in (0, 250, 500, 1000)]
    else:
        train_sets = [
            "train_real_real",   # C0: baseline (from your original build)
            "train_mixed",       # replacement: half real-fake, half synthetic
            "train_real_syn",    # C1: all synthetic fake
            "train_augmented",   # Option 1
            "train_lowres_real", # Option 3 baseline
            "train_lowres_aug",  # Option 3 augmented
            "train_c2_synreal_realfake",  # C2: synthetic-real + real-fake
            "train_c3_synreal_synfake",   # C3: synthetic-real + synthetic-fake
            "train_c6_full_augmented",    # C6: (RR+SR) + (RF+SF)
        ]

    all_rows = []
    for ts in train_sets:
        if not (cfg.PROCESSED_DIR / f"{ts}.csv").exists():
            print(f"  (skip {ts} — not built)")
            continue
        print(f"Training on {ts} ...")
        all_rows.extend(train_eval(ts, tests))

    df = pd.DataFrame(all_rows)
    out = EXTRA_DIR / ("sweep_results.csv" if args.sweep else "augmentation_results.csv")
    df.to_csv(out, index=False)
    print("\n=== RESULTS ===")
    print(df.to_string(index=False))
    print(f"\nSaved to {out}")

    # ---- interpretation helper for Option 1 ----
    if not args.sweep:
        print("\n--- Option 1 read (SVM, crossdomain F1) ---")
        try:
            base = df[(df.train_set=="train_real_real") & (df.model=="SVM") & (df.test=="crossdomain")]["f1"].values[0]
            aug = df[(df.train_set=="train_augmented") & (df.model=="SVM") & (df.test=="crossdomain")]["f1"].values[0]
            verdict = "HELPS" if aug > base else "does not help"
            print(f"  real_real F1={base}  vs  augmented F1={aug}  ->  synthetic {verdict}")
        except (IndexError, KeyError):
            pass

    # ---- Option 4 plot ----
    if args.sweep and args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for (model, test), g in df.groupby(["model", "test"]):
            amounts = [int(s.split("_")[-1]) for s in g["train_set"]]
            plt.plot(amounts, g["f1"], marker="o", label=f"{model}-{test}")
        plt.xlabel("Number of synthetic samples added")
        plt.ylabel("F1")
        plt.title("Option 4: F1 vs amount of synthetic augmentation")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        fig = EXTRA_DIR / "sweep_curve.png"
        plt.savefig(fig, dpi=150)
        print(f"Sweep curve saved to {fig}")


if __name__ == "__main__":
    main()
