
import pandas as pd
from sklearn.model_selection import train_test_split

import config as cfg


def load(name):
    return pd.read_csv(cfg.PROCESSED_DIR / f"{name}.csv")


def main():
    isot_real = load("isot_real")
    isot_fake = load("isot_fake")
    liar_fake = load("liar_fake")
    syn_path = cfg.SYNTHETIC_DIR / "synthetic_fake.csv"
    if not syn_path.exists():
        raise FileNotFoundError("Run generate_synthetic_fake.py first.")
    synthetic = pd.read_csv(syn_path)[["text", "label", "source"]]

    # ---- Split ISOT real into train pool + held-out test (80/20) ----
    real_train, real_test = train_test_split(
        isot_real, test_size=cfg.TEST_SIZE,
        random_state=cfg.SEED, shuffle=True,
    )
    print(f"ISOT real -> train {len(real_train):,} / test {len(real_test):,}")

    # ---- Shared test set: held-out real + ALL liar fake ----
    test = pd.concat([
        real_test[["text", "label", "source"]],
        liar_fake[["text", "label", "source"]],
    ], ignore_index=True)
    test = test.sample(frac=1.0, random_state=cfg.SEED).reset_index(drop=True)
    test.to_csv(cfg.PROCESSED_DIR / "test_shared.csv", index=False)
    print(f"Shared test set: {len(test):,} "
          f"({(test.label==cfg.LABEL_REAL).sum()} real / "
          f"{(test.label==cfg.LABEL_FAKE).sum()} fake)")

    n_real = len(real_train)
    # Keep fake count = min(real_train, available isot fake not in test, synthetic).
    # IMPORTANT: this must be the SAME target count for all three compositions,
    # otherwise real_real/mixed vs real_syn aren't comparable (real_syn is 100%
    # synthetic, so it needs `len(synthetic)` fake examples available, not
    # `len(synthetic) * 2` — that x2 only makes sense for the 50/50 "mixed" case
    # and was silently giving real_syn half the fake count of the other two).
    isot_fake_pool = isot_fake.sample(frac=1.0, random_state=cfg.SEED)
    n_fake = min(n_real, len(isot_fake_pool), len(synthetic))
    print(f"Using {n_fake:,} fake samples per composition "
          f"(same count for real_real / mixed / real_syn — capped by synthetic "
          f"supply: {len(synthetic):,} available).")

    # Cap the REAL side to n_fake too (balanced 1:1), instead of using the
    # full real_train (~17k). Uncapped, real_real/mixed/real_syn end up at a
    # ~34:1 real:fake ratio while the C2/C3 synthetic-real controls (built in
    # 04d) are 1:1 -- an uncontrolled second variable (class balance) riding
    # alongside the one you actually want to isolate (fake-class source).
    # error_analysis.py's confusion matrices showed LR/SVM at 34:1 collapse to
    # predicting ~100% "real" regardless of what the fake class contains, which
    # was masquerading as "synthetic fake breaks generalization." Capping here
    # makes real_real/mixed/real_syn/C2/C3 all comparable at the same ratio.
    real_part = real_train.head(n_fake)[["text", "label", "source"]]

    def assemble(fake_part, comp):
        df = pd.concat([real_part, fake_part], ignore_index=True)
        df = df.sample(frac=1.0, random_state=cfg.SEED).reset_index(drop=True)
        df.to_csv(cfg.PROCESSED_DIR / f"train_{comp}.csv", index=False)
        print(f"  train_{comp}: {len(df):,} "
              f"({(df.label==0).sum()} real / {(df.label==1).sum()} fake)")

    # real_real
    assemble(isot_fake_pool.head(n_fake)[["text", "label", "source"]], "real_real")

    # real_syn
    syn_fake = synthetic.head(n_fake)
    assemble(syn_fake[["text", "label", "source"]], "real_syn")

    # mixed (50/50)
    half = n_fake // 2
    mixed_fake = pd.concat([
        isot_fake_pool.head(half)[["text", "label", "source"]],
        synthetic.head(half)[["text", "label", "source"]],
    ], ignore_index=True)
    assemble(mixed_fake, "mixed")

    print("\nAll training sets + shared test set built.")


if __name__ == "__main__":
    main()
