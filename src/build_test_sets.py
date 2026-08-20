
"""Build the in-domain, LIAR and WELFake test sets, with ISOT overlap removed."""

import re

import pandas as pd
from sklearn.model_selection import train_test_split

import config as cfg


def load(name):
    return pd.read_csv(cfg.PROCESSED_DIR / f"{name}.csv")


def _norm(s):
    """Whitespace/case-insensitive key for corpus-overlap matching."""
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def main():
    isot_real = load("isot_real")
    isot_fake = load("isot_fake")
    liar_fake = load("liar_fake")

    _, real_test = train_test_split(
        isot_real, test_size=cfg.TEST_SIZE, random_state=cfg.SEED, shuffle=True
    )
    isot_fake_pool = isot_fake.sample(frac=1.0, random_state=cfg.SEED).reset_index(drop=True)
    n_fake_train = min(len(isot_real) - len(real_test), len(isot_fake_pool))
    isot_fake_heldout = isot_fake_pool.iloc[n_fake_train:]

    if len(isot_fake_heldout) == 0:
        _, isot_fake_heldout = train_test_split(
            isot_fake, test_size=cfg.TEST_SIZE, random_state=cfg.SEED, shuffle=True
        )
        print("Note: reused a fresh 20% ISOT-fake slice for the in-domain test.")

    cols = ["text", "label", "source"]

    indomain = pd.concat(
        [real_test[cols], isot_fake_heldout[cols]], ignore_index=True
    ).sample(frac=1.0, random_state=cfg.SEED).reset_index(drop=True)
    indomain.to_csv(cfg.PROCESSED_DIR / "test_indomain.csv", index=False)
    print(f"test_indomain:    {len(indomain):,} "
          f"({(indomain.label==0).sum()} real / {(indomain.label==1).sum()} fake)")

    cross = pd.concat(
        [real_test[cols], liar_fake[cols]], ignore_index=True
    ).sample(frac=1.0, random_state=cfg.SEED).reset_index(drop=True)
    cross.to_csv(cfg.PROCESSED_DIR / "test_crossdomain.csv", index=False)
    print(f"test_crossdomain: {len(cross):,} "
          f"({(cross.label==0).sum()} real / {(cross.label==1).sum()} fake)")

    welfake_path = cfg.PROCESSED_DIR / "welfake_fake.csv"
    if welfake_path.exists():
        welfake_fake = load("welfake_fake")
        n = len(liar_fake)
        welfake_fake_sample = welfake_fake.sample(n=n, random_state=cfg.SEED)
        cross2 = pd.concat(
            [real_test[cols], welfake_fake_sample[cols]], ignore_index=True
        ).sample(frac=1.0, random_state=cfg.SEED).reset_index(drop=True)
        cross2.to_csv(cfg.PROCESSED_DIR / "test_crossdomain2.csv", index=False)
        print(f"test_crossdomain2 (WELFake): {len(cross2):,} "
              f"({(cross2.label==0).sum()} real / {(cross2.label==1).sum()} fake)")

        isot_keys = set(isot_real["text"].map(_norm)) | set(isot_fake["text"].map(_norm))
        wf = welfake_fake.copy()
        wf["_key"] = wf["text"].map(_norm)
        clean_pool = (wf[~wf["_key"].isin(isot_keys)]
                      .drop_duplicates(subset="_key")
                      .drop(columns="_key"))
        removed = len(welfake_fake) - len(clean_pool)
        print(f"  WELFake fake pool: {len(welfake_fake):,} -> {len(clean_pool):,} clean "
              f"({removed:,} removed = {100*removed/len(welfake_fake):.1f}%: "
              f"ISOT-overlapping or internally duplicated)")

        if len(clean_pool) >= n:
            clean_sample = clean_pool.sample(n=n, random_state=cfg.SEED)
        else:
            clean_sample = clean_pool
            print(f"  (clean pool smaller than {n:,}; using all {len(clean_pool):,})")
        cross3 = pd.concat(
            [real_test[cols], clean_sample[cols]], ignore_index=True
        ).sample(frac=1.0, random_state=cfg.SEED).reset_index(drop=True)
        cross3.to_csv(cfg.PROCESSED_DIR / "test_crossdomain2_clean.csv", index=False)
        print(f"test_crossdomain2_clean:     {len(cross3):,} "
              f"({(cross3.label==0).sum()} real / {(cross3.label==1).sum()} fake)")
    else:
        print(f"  (skipping test_crossdomain2 -- {welfake_path} not found, "
              f"run load_data.py with WELFake_Dataset.csv in data/raw/ first)")

    print("\nDone. Use these test sets in run_lr_svm_extra_experiments.py / evaluate.py.")


if __name__ == "__main__":
    main()
