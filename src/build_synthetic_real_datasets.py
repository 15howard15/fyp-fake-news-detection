
"""Build the C2/C3 authorship-control compositions from synthetic-real news."""
import argparse
import difflib

import pandas as pd
from sklearn.model_selection import train_test_split

import config as cfg


def load(name):
    return pd.read_csv(cfg.PROCESSED_DIR / f"{name}.csv")


def mean_similarity(df, window=4000):
    """Mean difflib similarity between each row and its source article."""
    sims = [difflib.SequenceMatcher(None, str(s)[:window], str(t),
                                    autojunk=False).ratio()
            for s, t in zip(df["source_text"], df["text"])]
    return sum(sims) / max(len(sims), 1)


def check_symmetry(real_df, fake_df, tolerance_pp, fail):
    """Gate the SYMMETRIC pair on the two classes having been edited equally."""
    r, f = mean_similarity(real_df), mean_similarity(fake_df)
    gap = abs(r - f) * 100
    print(f"\n--- edit-distance symmetry check ---")
    print(f"  synthetic-real similarity to source: {r:.4f}")
    print(f"  synthetic-fake similarity to source: {f:.4f}")
    print(f"  gap: {gap:.1f} percentage points (tolerance {tolerance_pp})")
    if gap > tolerance_pp:
        msg = (f"Edit-distance asymmetry is {gap:.1f}pp, above the {tolerance_pp}pp "
               f"tolerance. The two classes were rewritten by different amounts, so "
               f"a model can use rewrite depth as a proxy for the label -- which is "
               f"the authorship shortcut these compositions exist to remove. "
               f"Regenerate with generate_synthetic_fake.py --symmetric.")
        if fail:
            raise SystemExit(f"\nFAIL: {msg}")
        print(f"  WARNING: {msg}")
    else:
        print(f"  OK -- within tolerance.")
    return {"real": r, "fake": f, "gap_pp": gap}


def save(df, name):
    df = df.sample(frac=1.0, random_state=cfg.SEED).reset_index(drop=True)
    df.to_csv(cfg.PROCESSED_DIR / f"{name}.csv", index=False)
    n_real = (df.label == cfg.LABEL_REAL).sum()
    n_fake = (df.label == cfg.LABEL_FAKE).sum()
    print(f"  {name}: {len(df):,}  ({n_real} real / {n_fake} fake)")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tolerance", type=float, default=5.0,
                     help="percentage points of mean source-similarity the two "
                          "classes of the SYMMETRIC pair may differ by (default 5)")
    ap.add_argument("--strict", action="store_true",
                     help="stop the build if the symmetry check breaches the "
                          "tolerance, instead of warning and continuing")
    args = ap.parse_args()

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

    real_train, _test = train_test_split(
        isot_real, test_size=cfg.TEST_SIZE, random_state=cfg.SEED, shuffle=True
    )
    isot_fake_pool = isot_fake.sample(frac=1.0, random_state=cfg.SEED).reset_index(drop=True)

    print("\n--- C2: synthetic-real + real-fake ---")
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
    LOWRES_N = 1000
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

    sym_fake_path = cfg.SYNTHETIC_DIR / "synthetic_fake_sym.csv"
    sym_real_path = cfg.SYNTHETIC_DIR / "synthetic_real_sym.csv"
    missing = [p.name for p in (sym_fake_path, sym_real_path) if not p.exists()]
    if missing:
        print(f"\n(skipping C2'/C3' -- {', '.join(missing)} not found; run "
              f"generate_synthetic_fake.py --symmetric and "
              f"generate_synthetic_real.py --symmetric first)")
    else:
        sym_fake_full = pd.read_csv(sym_fake_path)
        sym_real_full = pd.read_csv(sym_real_path)
        check_symmetry(sym_real_full, sym_fake_full, args.tolerance, args.strict)

        sym_fake = sym_fake_full[["text", "label", "source"]]
        sym_real = sym_real_full[["text", "label", "source"]]
        print("\n--- C2': symmetric synthetic-real + real-fake ---")
        n = min(len(sym_real), len(isot_fake_pool), len(sym_fake))
        c2s = pd.concat([sym_real.head(n),
                         isot_fake_pool.head(n)[["text", "label", "source"]]],
                        ignore_index=True)
        save(c2s, "train_c2_sym")

        print("\n--- C3': symmetric synthetic-real + symmetric synthetic-fake ---")
        n3 = min(len(sym_real), len(sym_fake))
        c3s = pd.concat([sym_real.head(n3), sym_fake.head(n3)], ignore_index=True)
        save(c3s, "train_c3_sym")
        print(f"  (compare against train_c3_synreal_synfake at matched scale "
              f"n={n3:,} -- same real class, same manipulation strategies, "
              f"differing only in how heavily the fake class was rewritten)")

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
