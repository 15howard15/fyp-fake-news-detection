
"""
build_swap_sweep_datasets.py -- balance-CONTROLLED synthetic-fraction sweep.

This replaces an earlier "augmentation" design that ADDED synthetic fake on top
of a fixed real-fake baseline, growing the fake class without growing the real
class -- 500 real vs 1,000 fake, a 1:2 imbalance. Confusion matrices showed
LR/SVM/BERT collapsing to ~100% recall / ~57% precision cross-domain under that
design, almost exactly what a trivial "always predict fake" classifier would
score: the same majority-class-collapse mechanism already fixed on the
replacement axis, re-introduced in the opposite direction. That design was
dropped and its scripts removed; the balance-controlled sweep below is what
replaced it.

This script instead builds a sweep that keeps the TOTAL fake count fixed at
500 (matching the 500 real news rows) and only varies what FRACTION of that
500 is synthetic vs real-fake. That isolates "does synthetic content help"
from "does adding synthetic unbalance the classes".

    swap_000 (  0% synthetic) = train_real_real  (already built, build_core_datasets.py)
    swap_025 ( 25% synthetic) = 375 real-fake + 125 synthetic-fake  [NEW]
    swap_050 ( 50% synthetic) = train_mixed      (already built, build_core_datasets.py)
    swap_075 ( 75% synthetic) = 125 real-fake + 375 synthetic-fake  [NEW]
    swap_100 (100% synthetic) = train_real_syn   (already built, build_core_datasets.py)

Only swap_025 and swap_075 need building here; the other three are re-used
by name in run_swap_sweep_experiment.py. All five draw from the EXACT same real_part
(same 500 rows, same seed) and the same isot_fake_pool / synthetic ordering
as build_core_datasets.py, so they differ from each other by ONLY the
real-fake/synthetic-fake split -- nothing else moves.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

import config as cfg

FRACTIONS = [0.25, 0.75]  # 0.00/0.50/1.00 already exist as real_real/mixed/real_syn


def main():
    isot_real = pd.read_csv(cfg.PROCESSED_DIR / "isot_real.csv")
    isot_fake = pd.read_csv(cfg.PROCESSED_DIR / "isot_fake.csv")
    synthetic = pd.read_csv(cfg.SYNTHETIC_DIR / "synthetic_fake.csv")[["text", "label", "source"]]

    real_train, _test = train_test_split(
        isot_real, test_size=cfg.TEST_SIZE, random_state=cfg.SEED, shuffle=True
    )
    isot_fake_pool = isot_fake.sample(frac=1.0, random_state=cfg.SEED)

    n_fake = min(len(real_train), len(isot_fake_pool), len(synthetic))  # 500 -- same cap as 04
    real_part = real_train.head(n_fake)[["text", "label", "source"]]

    print(f"Fixed total fake count: {n_fake} (matches real class: {len(real_part)})")

    for frac in FRACTIONS:
        n_syn = round(n_fake * frac)
        n_real_fake = n_fake - n_syn
        fake_part = pd.concat([
            isot_fake_pool.head(n_real_fake)[["text", "label", "source"]],
            synthetic.head(n_syn),
        ], ignore_index=True)
        df = pd.concat([real_part, fake_part], ignore_index=True)
        df = df.sample(frac=1.0, random_state=cfg.SEED).reset_index(drop=True)
        name = f"train_swap_{int(frac * 100):03d}"
        df.to_csv(cfg.PROCESSED_DIR / f"{name}.csv", index=False)
        n_r = (df.label == cfg.LABEL_REAL).sum()
        n_f = (df.label == cfg.LABEL_FAKE).sum()
        print(f"  {name}: {len(df):,}  ({n_r} real / {n_f} fake -- "
              f"{n_real_fake} real-fake + {n_syn} synthetic-fake)")

    print("\nDone. Full sweep (5 points, all balanced 500/500):")
    print("  train_swap_000 = train_real_real, train_swap_050 = train_mixed, "
          "train_swap_100 = train_real_syn (reuse existing files by these names "
          "in run_swap_sweep_experiment.py)")


if __name__ == "__main__":
    main()
