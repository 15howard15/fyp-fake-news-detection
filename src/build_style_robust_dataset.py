
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
