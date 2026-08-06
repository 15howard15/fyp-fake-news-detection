
import pandas as pd
from sklearn.model_selection import train_test_split

import config as cfg

# How many real-fake samples count as "low resource" (Option 3)
LOWRES_N = 1000
# Synthetic amounts for the sweep (Option 4). Capped by how many you generated.
SWEEP_AMOUNTS = [0, 250, 500, 1000]


def load(name):
    return pd.read_csv(cfg.PROCESSED_DIR / f"{name}.csv")


def save(df, name):
    df = df.sample(frac=1.0, random_state=cfg.SEED).reset_index(drop=True)
    df.to_csv(cfg.PROCESSED_DIR / f"{name}.csv", index=False)
    n_real = (df.label == cfg.LABEL_REAL).sum()
    n_fake = (df.label == cfg.LABEL_FAKE).sum()
    print(f"  {name}: {len(df):,}  ({n_real} real / {n_fake} fake)")


def main():
    isot_real = load("isot_real")
    isot_fake = load("isot_fake")

    syn_path = cfg.SYNTHETIC_DIR / "synthetic_fake.csv"
    if not syn_path.exists():
        raise FileNotFoundError("Run generate_synthetic_fake.py first.")
    synthetic = pd.read_csv(syn_path)[["text", "label", "source"]]
    print(f"Loaded {len(synthetic)} synthetic samples.")

    # ---- Recreate the SAME train/test split used in build_core_datasets.py ----
    # (same SEED + TEST_SIZE => identical split, so results stay comparable)
    real_train, _test = train_test_split(
        isot_real, test_size=cfg.TEST_SIZE, random_state=cfg.SEED, shuffle=True
    )
    isot_fake_pool = isot_fake.sample(frac=1.0, random_state=cfg.SEED).reset_index(drop=True)
    real_part = real_train[["text", "label", "source"]]
    syn_part = synthetic[["text", "label", "source"]]

    print("\n--- OPTION 1: Augmentation (controlled) ---")
    # CONTROLLED comparison: load the EXACT train_real_real.csv built by 04, and
    # only ADD synthetic on top. This guarantees train_augmented differs from
    # train_real_real by ONLY the synthetic samples -- no accidental change in
    # real-fake volume or class balance. Report THIS pair for "does
    # augmentation help", not a comparison against a differently-sized baseline.
    real_real_path = cfg.PROCESSED_DIR / "train_real_real.csv"
    if not real_real_path.exists():
        raise FileNotFoundError("Run build_core_datasets.py first (need train_real_real.csv).")
    real_real = pd.read_csv(real_real_path)[["text", "label", "source"]]
    n_fake_controlled = int((real_real.label == cfg.LABEL_FAKE).sum())
    augmented = pd.concat([real_real, syn_part], ignore_index=True)
    save(augmented, "train_augmented")
    print(f"  (train_augmented = train_real_real [{n_fake_controlled:,} real-fake] "
          f"+ {len(syn_part):,} synthetic on top. Compare vs train_real_real.)")

    # OPTIONAL / separate: an "at-scale" variant using the FULL real-fake pool.
    # NOT a controlled comparison against train_real_real (real-fake count is
    # much larger here) -- it answers a different question ("does a little
    # synthetic data help when you already have abundant real fake news?").
    # Keep it labeled separately and never compare its F1 directly to
    # train_real_real as if it were the same experiment.
    n_fake_full = min(len(real_part), len(isot_fake_pool))
    base_real_fake_full = isot_fake_pool.head(n_fake_full)[["text", "label", "source"]]
    augmented_full = pd.concat([real_part, base_real_fake_full, syn_part], ignore_index=True)
    save(augmented_full, "train_augmented_full")
    print(f"  (train_augmented_full uses {n_fake_full:,} real-fake -- NOT comparable "
          "to train_real_real. Report as a separate 'at-scale' condition.)")

    print("\n--- OPTION 3: Low-resource ---")
    small_fake = isot_fake_pool.head(LOWRES_N)[["text", "label", "source"]]
    # keep real count balanced-ish to the small fake set
    small_real = real_part.head(max(LOWRES_N, len(small_fake)))
    lowres_real = pd.concat([small_real, small_fake], ignore_index=True)
    save(lowres_real, "train_lowres_real")
    lowres_aug = pd.concat([small_real, small_fake, syn_part], ignore_index=True)
    save(lowres_aug, "train_lowres_aug")
    print("  (compare train_lowres_aug vs train_lowres_real -- already controlled: "
          "real-fake count fixed, only synthetic added on top)")

    print("\n--- OPTION 4: Synthetic-amount sweep ---")
    # Sweep is anchored to the SAME train_real_real baseline (not the full pool),
    # so sweep_0000 == train_real_real exactly, and the curve reads as
    # "starting from your reported baseline, how does F1 change as you add
    # increasing amounts of synthetic data on top?"
    for amt in SWEEP_AMOUNTS:
        amt_capped = min(amt, len(syn_part))
        sweep = pd.concat([real_real, syn_part.head(amt_capped)], ignore_index=True)
        save(sweep, f"train_sweep_{amt:04d}")
    print("  (plot F1 across train_sweep_* to show the augmentation curve; "
          "train_sweep_0000 should be identical to train_real_real)")

    print("\nDone. New training sets are in data/processed/.")
    print("Test set is unchanged -- reuse test_shared.csv for cross-domain,")
    print("and 04c builds test_indomain / test_crossdomain separately.")


if __name__ == "__main__":
    main()
