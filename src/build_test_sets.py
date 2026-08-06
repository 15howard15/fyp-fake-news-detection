
import pandas as pd
from sklearn.model_selection import train_test_split

import config as cfg


def load(name):
    return pd.read_csv(cfg.PROCESSED_DIR / f"{name}.csv")


def main():
    isot_real = load("isot_real")
    isot_fake = load("isot_fake")
    liar_fake = load("liar_fake")

    # Same split as the training build so the held-out portions match exactly.
    _, real_test = train_test_split(
        isot_real, test_size=cfg.TEST_SIZE, random_state=cfg.SEED, shuffle=True
    )
    # Hold out ISOT fake too, using the SAME logic as the training build so we
    # don't leak training fakes into the in-domain test.
    isot_fake_pool = isot_fake.sample(frac=1.0, random_state=cfg.SEED).reset_index(drop=True)
    n_fake_train = min(len(isot_real) - len(real_test), len(isot_fake_pool))
    isot_fake_heldout = isot_fake_pool.iloc[n_fake_train:]  # everything not used in training

    if len(isot_fake_heldout) == 0:
        # If all ISOT fake was used in training, carve a fresh 20% held-out slice.
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

    # Second cross-domain target: WELFake instead of LIAR. Genre-matched to
    # ISOT (full-length articles) unlike LIAR (short political statements),
    # so this isolates topic/source domain shift from the length/genre shift
    # LIAR conflates it with. Uses the SAME real_test rows as test_crossdomain
    # (only the fake-class source differs) and samples WELFake-fake down to
    # len(liar_fake) so both cross-domain test sets are the same size and
    # directly comparable.
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
    else:
        print(f"  (skipping test_crossdomain2 -- {welfake_path} not found, "
              f"run load_data.py with WELFake_Dataset.csv in data/raw/ first)")

    print("\nDone. Use these test sets in run_lr_svm_extra_experiments.py / evaluate.py.")


if __name__ == "__main__":
    main()
