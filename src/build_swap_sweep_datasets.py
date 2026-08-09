
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
