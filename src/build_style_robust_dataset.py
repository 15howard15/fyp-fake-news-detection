
"""
build_style_robust_dataset.py -- builds train_style_robust: the actual
test of Objective 4's hypothesis, done properly. Unlike `mixed` (which adds
generic synthetic fake news, not built for style-robustness), this adds
PAIRED counter-style twins of articles already in train_real_real -- the
same content in both tones, same true label -- so the model can no longer
use tone as a shortcut between the classes.

    train_style_robust = train_real_real (500 real / 500 fake)
                        + counter_style_training.csv (100 sensationalized-real
                          + 100 neutralized-fake, twins of articles already
                          in train_real_real)
                        = 600 real / 600 fake -- still balanced, so this
                          isn't confounded by the imbalance issue found
                          earlier in the project.

Run AFTER generate_style.py counter-training.
"""
import pandas as pd

import config as cfg


def main():
    real_real = pd.read_csv(cfg.PROCESSED_DIR / "train_real_real.csv")[["text", "label", "source"]]
    counter = pd.read_csv(cfg.SYNTHETIC_DIR / "counter_style_training.csv")[["text", "label", "source"]]

    df = pd.concat([real_real, counter], ignore_index=True)
    df = df.sample(frac=1.0, random_state=cfg.SEED).reset_index(drop=True)
    out = cfg.PROCESSED_DIR / "train_style_robust.csv"
    df.to_csv(out, index=False)

    n_real = (df.label == cfg.LABEL_REAL).sum()
    n_fake = (df.label == cfg.LABEL_FAKE).sum()
    print(f"train_style_robust: {len(df):,} ({n_real} real / {n_fake} fake) -> {out}")


if __name__ == "__main__":
    main()
