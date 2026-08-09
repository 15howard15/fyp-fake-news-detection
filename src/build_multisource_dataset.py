
"""
build_multisource_dataset.py -- Objective 1 fix: builds a
"real_syn_multisource" composition where the synthetic fake class is drawn
from TWO source datasets (ISOT via generate_synthetic_fake.py, LIAR via
generate_synthetic_fake_liar.py) instead of one.

Everything else -- the real class, and the TOTAL fake count -- is kept
identical to the existing train_real_syn.csv, so this is a direct,
single-variable ablation: same size, same real class, only the source mix
of the synthetic fake class changes. Compare model performance on this
composition against train_real_syn.csv to test whether "diverse sources"
(Objective 1) actually changes detection performance.
"""
import pandas as pd

import config as cfg


def main():
    real_syn_path = cfg.PROCESSED_DIR / "train_real_syn.csv"
    if not real_syn_path.exists():
        raise FileNotFoundError("Run build_core_datasets.py first.")
    existing = pd.read_csv(real_syn_path)
    real_part = existing[existing.label == cfg.LABEL_REAL].reset_index(drop=True)
    n_fake = int((existing.label == cfg.LABEL_FAKE).sum())
    print(f"train_real_syn.csv: {len(real_part):,} real / {n_fake:,} fake "
          f"(ISOT-sourced synthetic only)")

    liar_syn_path = cfg.SYNTHETIC_DIR / "synthetic_fake_liar.csv"
    if not liar_syn_path.exists():
        raise FileNotFoundError("Run generate_synthetic_fake_liar.py first.")
    liar_synthetic = pd.read_csv(liar_syn_path)[["text", "label", "source"]]

    isot_syn_path = cfg.SYNTHETIC_DIR / "synthetic_fake.csv"
    isot_synthetic = pd.read_csv(isot_syn_path)[["text", "label", "source"]]

    # Replace up to half of the fake class with LIAR-sourced synthetic,
    # keeping the TOTAL fake count identical to train_real_syn.csv so this is
    # a single-variable swap (source mix), not a size or balance change.
    target_liar = n_fake // 2
    n_liar = min(target_liar, len(liar_synthetic))
    n_isot = n_fake - n_liar
    if n_liar < target_liar:
        print(f"NOTE: only {len(liar_synthetic)} LIAR-sourced synthetic samples "
              f"available, using all of them ({n_liar}) instead of the target "
              f"{target_liar}. Run generate_synthetic_fake_liar.py with a "
              f"higher --n to get closer to a 50/50 source split.")
    if n_isot > len(isot_synthetic):
        raise ValueError(
            f"Need {n_isot} ISOT-sourced synthetic rows but only "
            f"{len(isot_synthetic)} exist in {isot_syn_path}."
        )

    fake_part = pd.concat([
        isot_synthetic.head(n_isot),
        liar_synthetic.head(n_liar),
    ], ignore_index=True)

    df = pd.concat([real_part, fake_part], ignore_index=True)
    df = df.sample(frac=1.0, random_state=cfg.SEED).reset_index(drop=True)
    out_path = cfg.PROCESSED_DIR / "train_real_syn_multisource.csv"
    df.to_csv(out_path, index=False)
    print(f"train_real_syn_multisource: {len(df):,} "
          f"({(df.label==cfg.LABEL_REAL).sum()} real / "
          f"{(df.label==cfg.LABEL_FAKE).sum()} fake, of which {n_isot} "
          f"ISOT-sourced synthetic + {n_liar} LIAR-sourced synthetic)")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
