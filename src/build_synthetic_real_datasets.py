
"""
build_synthetic_real_datasets.py -- build the C2/C3/C6 control compositions
using synthetic-REAL news (from generate_synthetic_real.py).

Together with train_real_real (C0) and train_real_syn (C1, from build_core_
datasets.py), this completes the 2x2 "replacement" matrix from your notes:

              real-fake (RF)     synthetic-fake (SF)
    real-real (RR)   C0              C1
    synth-real (SR)  C2              C3

C0 (train_real_real) and C1 (train_real_syn) already exist. This script adds:
    C2 = synthetic_real + real_fake      -> train_c2_synreal_realfake.csv
    C3 = synthetic_real + synthetic_fake -> train_c3_synreal_synfake.csv

Reading C2 vs C0 isolates the effect of paraphrasing the REAL side only.
Reading C3 vs C1 isolates the same thing when the fake side is ALSO synthetic.
If C3's F1 collapses even though its fake-class content is exactly the same
distortions as C1, but C2 (same real side as C3, real fake-class) does NOT
collapse, that's evidence the model is reacting to something about having
BOTH classes machine-authored (an authorship-detection shortcut) rather than
genuine content-based fakeness detection.

This script also adds C6, the "augment both sides" condition from your notes
("real real + synthetic real + real fake + synthetic fake" -- resolved to
include real-fake, since without it the fake class would be 100% synthetic,
which isn't augmentation in the "add on top of a fixed baseline" sense):
    C6 = (real_real + synthetic_real) + (real_fake + synthetic_fake)
       -> train_c6_full_augmented.csv
Built at the SAME 1,000/1,000 scale as train_lowres_real / train_lowres_aug,
so the chain lowres_real -> lowres_aug -> C6 isolates one change at a time:
first add synthetic fake news, then ALSO diversify the real side with
synthetic-real.

Run AFTER generate_synthetic_real.py has produced data/synthetic/synthetic_real.csv.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

import config as cfg


def load(name):
    return pd.read_csv(cfg.PROCESSED_DIR / f"{name}.csv")


def save(df, name):
    df = df.sample(frac=1.0, random_state=cfg.SEED).reset_index(drop=True)
    df.to_csv(cfg.PROCESSED_DIR / f"{name}.csv", index=False)
    n_real = (df.label == cfg.LABEL_REAL).sum()
    n_fake = (df.label == cfg.LABEL_FAKE).sum()
    print(f"  {name}: {len(df):,}  ({n_real} real / {n_fake} fake)")
    return df


def main():
    syn_real_path = cfg.SYNTHETIC_DIR / "synthetic_real.csv"
    if not syn_real_path.exists():
        raise FileNotFoundError(
            "Run generate_synthetic_real.py first (need data/synthetic/synthetic_real.csv)."
        )
    syn_real = pd.read_csv(syn_real_path)[["text", "label", "source"]]
    print(f"Loaded {len(syn_real):,} synthetic-real articles.")

    isot_real = load("isot_real")
    isot_fake = load("isot_fake")
    syn_fake_path = cfg.SYNTHETIC_DIR / "synthetic_fake.csv"
    if not syn_fake_path.exists():
        raise FileNotFoundError("Run generate_synthetic_fake.py first (need synthetic_fake.csv).")
    syn_fake = pd.read_csv(syn_fake_path)[["text", "label", "source"]]

    # Recreate the SAME split as everywhere else (same SEED), so real_train
    # here matches the train portion used by every other script -- no leakage
    # into real_test / test_indomain / test_crossdomain.
    real_train, _test = train_test_split(
        isot_real, test_size=cfg.TEST_SIZE, random_state=cfg.SEED, shuffle=True
    )
    isot_fake_pool = isot_fake.sample(frac=1.0, random_state=cfg.SEED).reset_index(drop=True)

    print("\n--- C2: synthetic-real + real-fake ---")
    # Capped by len(syn_fake) too (not just syn_real/isot_fake_pool) so C2
    # lands at the SAME n as C0/C1/mixed/C3 (500/500, set by synthetic_fake
    # supply) instead of 1,000/1,000. Without this, the 5-way replacement
    # comparison had two conditions at a different scale AND balance than the
    # other three -- see the comment in build_core_datasets.py for why that
    # confound mattered (it was producing a spurious "collapse" reading).
    n_c2 = min(len(syn_real), len(isot_fake_pool), len(syn_fake))
    real_fake_part = isot_fake_pool.head(n_c2)[["text", "label", "source"]]
    c2 = pd.concat([syn_real.head(n_c2), real_fake_part], ignore_index=True)
    save(c2, "train_c2_synreal_realfake")
    print(f"  (n={n_c2:,}/{n_c2:,} -- matched scale to train_real_real / "
          "train_mixed / train_real_syn / C3 for a fair 5-way comparison.)")

    print("\n--- C3: synthetic-real + synthetic-fake ---")
    n_c3 = min(len(syn_real), len(syn_fake))
    c3 = pd.concat([syn_real.head(n_c3), syn_fake.head(n_c3)], ignore_index=True)
    save(c3, "train_c3_synreal_synfake")
    print(f"  (compare vs train_real_syn at matched scale n={n_c3:,})")

    print("\n--- C6: (real_real + synthetic_real) + (real_fake + synthetic_fake) ---")
    LOWRES_N = 1000  # matches train_lowres_real / train_lowres_aug scale
    real_part_1000 = real_train.head(LOWRES_N)[["text", "label", "source"]]
    fake_part_1000 = isot_fake_pool.head(LOWRES_N)[["text", "label", "source"]]
    n_sr = min(len(syn_real), LOWRES_N)
    c6_real = pd.concat([real_part_1000, syn_real.head(n_sr)], ignore_index=True)
    c6_fake = pd.concat([fake_part_1000, syn_fake], ignore_index=True)
    c6 = pd.concat([c6_real, c6_fake], ignore_index=True)
    save(c6, "train_c6_full_augmented")
    print(f"  (real class = {len(real_part_1000):,} real-real + {n_sr:,} synthetic-real; "
          f"fake class = {len(fake_part_1000):,} real-fake + {len(syn_fake):,} synthetic-fake. "
          "Compare vs train_lowres_aug (same RR/RF/SF, no SR) to isolate the added "
          "synthetic-real effect specifically.)")

    print("\nDone.")
    print("For a fair C0-vs-C2 and C1-vs-C3 read, evaluate all four "
          "(train_real_real, train_real_syn, train_c2_synreal_realfake, "
          "train_c3_synreal_synfake) on the SAME test set (test_crossdomain / "
          "test_shared) and report counts alongside F1 -- these sets are only "
          "as large as your synthetic-real supply, so they will likely be "
          "smaller than the full train_real_real/train_real_syn unless you "
          "generate enough synthetic-real articles to match. "
          "For C6, evaluate on BOTH test_indomain and test_crossdomain, alongside "
          "train_lowres_real and train_lowres_aug, to extend the augmentation chain.")


if __name__ == "__main__":
    main()
