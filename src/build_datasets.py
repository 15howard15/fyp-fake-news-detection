"""build_datasets.py -- assembles every training composition from the raw pools."""
import argparse
import difflib
import re

import pandas as pd
from sklearn.model_selection import train_test_split

import config as cfg
from generate_synthetic_fake import LENGTH_SPECS

FRACTIONS = [0.25, 0.75]


def isot_pools():
    """The one canonical ISOT split, shared by every composition."""
    isot_real = pd.read_csv(cfg.PROCESSED_DIR / "isot_real.csv")
    isot_fake = pd.read_csv(cfg.PROCESSED_DIR / "isot_fake.csv")
    real_train, real_test = train_test_split(
        isot_real, test_size=cfg.TEST_SIZE, random_state=cfg.SEED, shuffle=True)
    isot_fake_pool = isot_fake.sample(frac=1.0, random_state=cfg.SEED)
    return real_train, real_test, isot_fake_pool


def _shuffled(df):
    return df.sample(frac=1.0, random_state=cfg.SEED).reset_index(drop=True)


def _write(df, name):
    df.to_csv(cfg.PROCESSED_DIR / f"{name}.csv", index=False)
    n_r = (df.label == cfg.LABEL_REAL).sum()
    n_f = (df.label == cfg.LABEL_FAKE).sum()
    print(f"  {name}: {len(df):,}  ({n_r} real / {n_f} fake)")
    return df


def truncate_to_sentences(text: str, target: int) -> str:
    """Cut a real article down to about `target` words, keeping whole sentences."""
    sents = [s for s in re.split(r"(?<=[.!?])\s+", str(text).strip()) if s]
    if not sents:
        return ""
    counts = [len(s.split()) for s in sents]
    best, best_err = None, None
    for i in range(len(sents)):
        total = 0
        for j in range(i, len(sents)):
            total += counts[j]
            err = abs(total - target)
            if best_err is None or err < best_err:
                best, best_err = (i, j), err
            if total >= target:
                break
    i, j = best
    return " ".join(sents[i:j + 1])


def cmd_core(args):
    """real_real / mixed / real_syn + the shared test set."""
    liar_fake = pd.read_csv(cfg.PROCESSED_DIR / "liar_fake.csv")
    syn_path = cfg.SYNTHETIC_DIR / "synthetic_fake.csv"
    if not syn_path.exists():
        raise FileNotFoundError("Run generate_synthetic_fake.py first.")
    synthetic = pd.read_csv(syn_path)[["text", "label", "source"]]

    real_train, real_test, isot_fake_pool = isot_pools()
    print(f"ISOT real -> train {len(real_train):,} / test {len(real_test):,}")

    test = pd.concat([
        real_test[["text", "label", "source"]],
        liar_fake[["text", "label", "source"]],
    ], ignore_index=True)
    test = _shuffled(test)
    test.to_csv(cfg.PROCESSED_DIR / "test_shared.csv", index=False)
    print(f"Shared test set: {len(test):,} "
          f"({(test.label==cfg.LABEL_REAL).sum()} real / "
          f"{(test.label==cfg.LABEL_FAKE).sum()} fake)")

    n_real = len(real_train)
    n_fake = min(n_real, len(isot_fake_pool), len(synthetic))
    print(f"Using {n_fake:,} fake samples per composition "
          f"(same count for real_real / mixed / real_syn — capped by synthetic "
          f"supply: {len(synthetic):,} available).")

    real_part = real_train.head(n_fake)[["text", "label", "source"]]

    def assemble(fake_part, comp):
        df = _shuffled(pd.concat([real_part, fake_part], ignore_index=True))
        return _write(df, f"train_{comp}")

    assemble(isot_fake_pool.head(n_fake)[["text", "label", "source"]], "real_real")
    assemble(synthetic.head(n_fake)[["text", "label", "source"]], "real_syn")

    half = n_fake // 2
    assemble(pd.concat([
        isot_fake_pool.head(half)[["text", "label", "source"]],
        synthetic.head(half)[["text", "label", "source"]],
    ], ignore_index=True), "mixed")

    _core_mixedlen()
    print("\nAll training sets + shared test set built.")


def _core_mixedlen():
    """real_syn_mixedlen: the same recipe with the length confound removed."""
    ml_path = cfg.SYNTHETIC_DIR / "synthetic_fake_mixedlen.csv"
    if not ml_path.exists():
        print(f"\n(skipping real_syn_mixedlen -- {ml_path.name} not found; "
              f"run generate_synthetic_fake.py --lengths short medium long)")
        return
    ml = pd.read_csv(ml_path)
    missing = {"length", "source_text"} - set(ml.columns)
    if missing:
        raise ValueError(f"{ml_path.name} is missing {missing}; regenerate it.")
    ml_rows = []
    for _, r in ml.iterrows():
        target = LENGTH_SPECS[r["length"]][0]
        ml_rows.append({"text": truncate_to_sentences(r["source_text"], target),
                        "label": cfg.LABEL_REAL, "source": "isot"})
        ml_rows.append({"text": r["text"], "label": cfg.LABEL_FAKE,
                        "source": "synthetic"})
    ml_df = _shuffled(pd.DataFrame(ml_rows))
    ml_df.to_csv(cfg.PROCESSED_DIR / "train_real_syn_mixedlen.csv", index=False)
    w = ml_df["text"].str.split().str.len()
    print(f"\n  train_real_syn_mixedlen: {len(ml_df):,} "
          f"({(ml_df.label==0).sum()} real / {(ml_df.label==1).sum()} fake)")
    print(f"    median words -- real {int(w[ml_df.label==0].median())} / "
          f"fake {int(w[ml_df.label==1].median())}")


def cmd_sweep(args):
    """The balance-CONTROLLED synthetic-fraction sweep."""
    synthetic = pd.read_csv(cfg.SYNTHETIC_DIR / "synthetic_fake.csv")[
        ["text", "label", "source"]]
    real_train, _test, isot_fake_pool = isot_pools()

    n_fake = min(len(real_train), len(isot_fake_pool), len(synthetic))
    real_part = real_train.head(n_fake)[["text", "label", "source"]]
    print(f"Fixed total fake count: {n_fake} (matches real class: {len(real_part)})")

    for frac in FRACTIONS:
        n_syn = round(n_fake * frac)
        n_real_fake = n_fake - n_syn
        fake_part = pd.concat([
            isot_fake_pool.head(n_real_fake)[["text", "label", "source"]],
            synthetic.head(n_syn),
        ], ignore_index=True)
        df = _shuffled(pd.concat([real_part, fake_part], ignore_index=True))
        _write(df, f"train_swap_{int(frac * 100):03d}")
        print(f"      ({n_real_fake} real-fake + {n_syn} synthetic-fake)")

    print("\nDone. Full sweep (5 points, all balanced 500/500): "
          "swap_000 = train_real_real, swap_050 = train_mixed, "
          "swap_100 = train_real_syn.")


def cmd_style_robust(args):
    """train_style_robust: the actual test of Objective 4's hypothesis."""
    cols = ["text", "label", "source"]
    real_real = pd.read_csv(cfg.PROCESSED_DIR / "train_real_real.csv")[cols]
    counter = pd.read_csv(cfg.SYNTHETIC_DIR / "counter_style_training.csv")[cols]
    df = _shuffled(pd.concat([real_real, counter], ignore_index=True))
    _write(df, "train_style_robust")


def cmd_multisource(args):
    """real_syn with half its synthetic fake class re-sourced from LIAR."""
    real_syn_path = cfg.PROCESSED_DIR / "train_real_syn.csv"
    if not real_syn_path.exists():
        raise FileNotFoundError("Run `build_datasets.py core` first.")
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
            f"{len(isot_synthetic)} exist in {isot_syn_path}.")

    fake_part = pd.concat([isot_synthetic.head(n_isot),
                           liar_synthetic.head(n_liar)], ignore_index=True)
    df = _shuffled(pd.concat([real_part, fake_part], ignore_index=True))
    _write(df, "train_real_syn_multisource")
    print(f"      ({n_isot} ISOT-sourced synthetic + {n_liar} LIAR-sourced)")


def _norm(s):
    """Whitespace/case-insensitive key for corpus-overlap matching."""
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def cmd_test_sets(args):
    """Build the in-domain, LIAR and WELFake test sets, with ISOT overlap removed."""
    isot_real = pd.read_csv(cfg.PROCESSED_DIR / "isot_real.csv")
    isot_fake = pd.read_csv(cfg.PROCESSED_DIR / "isot_fake.csv")
    liar_fake = pd.read_csv(cfg.PROCESSED_DIR / "liar_fake.csv")

    _, real_test, isot_fake_pool = isot_pools()
    isot_fake_pool = isot_fake_pool.reset_index(drop=True)
    n_fake_train = min(len(isot_real) - len(real_test), len(isot_fake_pool))
    isot_fake_heldout = isot_fake_pool.iloc[n_fake_train:]

    if len(isot_fake_heldout) == 0:
        _, isot_fake_heldout = train_test_split(
            isot_fake, test_size=cfg.TEST_SIZE, random_state=cfg.SEED, shuffle=True)
        print("Note: reused a fresh 20% ISOT-fake slice for the in-domain test.")

    cols = ["text", "label", "source"]

    indomain = _shuffled(pd.concat([real_test[cols], isot_fake_heldout[cols]],
                                   ignore_index=True))
    indomain.to_csv(cfg.PROCESSED_DIR / "test_indomain.csv", index=False)
    print(f"test_indomain:    {len(indomain):,} "
          f"({(indomain.label==0).sum()} real / {(indomain.label==1).sum()} fake)")

    cross = _shuffled(pd.concat([real_test[cols], liar_fake[cols]], ignore_index=True))
    cross.to_csv(cfg.PROCESSED_DIR / "test_crossdomain.csv", index=False)
    print(f"test_crossdomain: {len(cross):,} "
          f"({(cross.label==0).sum()} real / {(cross.label==1).sum()} fake)")

    welfake_path = cfg.PROCESSED_DIR / "welfake_fake.csv"
    if not welfake_path.exists():
        print(f"  (skipping test_crossdomain2 -- {welfake_path} not found, "
              f"run load_data.py with WELFake_Dataset.csv in data/raw/ first)")
        return

    welfake_fake = pd.read_csv(welfake_path)
    n = len(liar_fake)
    welfake_fake_sample = welfake_fake.sample(n=n, random_state=cfg.SEED)
    cross2 = _shuffled(pd.concat([real_test[cols], welfake_fake_sample[cols]],
                                 ignore_index=True))
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
    cross3 = _shuffled(pd.concat([real_test[cols], clean_sample[cols]],
                                 ignore_index=True))
    cross3.to_csv(cfg.PROCESSED_DIR / "test_crossdomain2_clean.csv", index=False)
    print(f"test_crossdomain2_clean:     {len(cross3):,} "
          f"({(cross3.label==0).sum()} real / {(cross3.label==1).sum()} fake)")


def _mean_similarity(df, window=4000):
    """Mean difflib similarity between each row and its source article."""
    sims = [difflib.SequenceMatcher(None, str(s)[:window], str(t),
                                    autojunk=False).ratio()
            for s, t in zip(df["source_text"], df["text"])]
    return sum(sims) / max(len(sims), 1)


def _check_symmetry(real_df, fake_df, tolerance_pp, fail):
    """Gate the symmetric pair on the two classes having been edited equally."""
    r, f = _mean_similarity(real_df), _mean_similarity(fake_df)
    gap = abs(r - f) * 100
    print("\n--- edit-distance symmetry check ---")
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
        print("  OK -- within tolerance.")
    return {"real": r, "fake": f, "gap_pp": gap}


def cmd_controls(args):
    """Build the C2/C3 authorship-control compositions from synthetic-real news."""
    syn_real_path = cfg.SYNTHETIC_DIR / "synthetic_real.csv"
    if not syn_real_path.exists():
        raise FileNotFoundError(
            "Run generate_synthetic_real.py first (need data/synthetic/synthetic_real.csv).")
    syn_real = pd.read_csv(syn_real_path)[["text", "label", "source"]]
    print(f"Loaded {len(syn_real):,} synthetic-real articles.")

    syn_fake_path = cfg.SYNTHETIC_DIR / "synthetic_fake.csv"
    if not syn_fake_path.exists():
        raise FileNotFoundError("Run generate_synthetic_fake.py first (need synthetic_fake.csv).")
    syn_fake = pd.read_csv(syn_fake_path)[["text", "label", "source"]]

    _real_train, _test, isot_fake_pool = isot_pools()
    isot_fake_pool = isot_fake_pool.reset_index(drop=True)

    print("\n--- C2: synthetic-real + real-fake ---")
    n_c2 = min(len(syn_real), len(isot_fake_pool), len(syn_fake))
    c2 = pd.concat([syn_real.head(n_c2),
                    isot_fake_pool.head(n_c2)[["text", "label", "source"]]],
                   ignore_index=True)
    _write(_shuffled(c2), "train_c2_synreal_realfake")
    print(f"  (n={n_c2:,}/{n_c2:,} -- matched scale to train_real_real / "
          "train_mixed / train_real_syn / C3 for a fair 5-way comparison.)")

    print("\n--- C3: synthetic-real + synthetic-fake ---")
    n_c3 = min(len(syn_real), len(syn_fake))
    c3 = pd.concat([syn_real.head(n_c3), syn_fake.head(n_c3)], ignore_index=True)
    _write(_shuffled(c3), "train_c3_synreal_synfake")
    print(f"  (compare vs train_real_syn at matched scale n={n_c3:,})")

    sym_fake_path = cfg.SYNTHETIC_DIR / "synthetic_fake_sym.csv"
    sym_real_path = cfg.SYNTHETIC_DIR / "synthetic_real_sym.csv"
    missing = [p.name for p in (sym_fake_path, sym_real_path) if not p.exists()]
    if missing:
        print(f"\n(skipping C2'/C3' -- {', '.join(missing)} not found; run "
              f"generate_synthetic_fake.py --symmetric and "
              f"generate_synthetic_real.py --symmetric first)")
        return

    sym_fake_full = pd.read_csv(sym_fake_path)
    sym_real_full = pd.read_csv(sym_real_path)
    _check_symmetry(sym_real_full, sym_fake_full, args.tolerance, args.strict)
    sym_fake = sym_fake_full[["text", "label", "source"]]
    sym_real = sym_real_full[["text", "label", "source"]]

    print("\n--- C2': symmetric synthetic-real + real-fake ---")
    n = min(len(sym_real), len(isot_fake_pool), len(sym_fake))
    c2s = pd.concat([sym_real.head(n),
                     isot_fake_pool.head(n)[["text", "label", "source"]]],
                    ignore_index=True)
    _write(_shuffled(c2s), "train_c2_sym")

    print("\n--- C3': symmetric synthetic-real + symmetric synthetic-fake ---")
    n3 = min(len(sym_real), len(sym_fake))
    c3s = pd.concat([sym_real.head(n3), sym_fake.head(n3)], ignore_index=True)
    _write(_shuffled(c3s), "train_c3_sym")
    print(f"  (compare against train_c3_synreal_synfake at matched scale "
          f"n={n3:,} -- same real class, same manipulation strategies, "
          f"differing only in how heavily the fake class was rewritten)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    for name, helptext in [
        ("core", "real_real / mixed / real_syn + shared test set (+ mixedlen)"),
        ("sweep", "swap_025 / swap_075, the synthetic-fraction sweep points"),
        ("style-robust", "train_real_real + counter-style twins"),
        ("multisource", "real_syn with half the fake class re-sourced from LIAR"),
        ("test-sets", "in-domain / LIAR / WELFake test sets, ISOT overlap removed"),
        ("controls", "C2/C3 and their edit-matched twins, the authorship controls"),
        ("all", "core then sweep, the two that must stay in step"),
    ]:
        p = sub.add_parser(name, help=helptext)
        if name == "controls":
            p.add_argument("--tolerance", type=float, default=5.0,
                           help="percentage points of mean source-similarity the two "
                                "classes of the symmetric pair may differ by")
            p.add_argument("--strict", action="store_true",
                           help="stop the build if the symmetry check breaches the "
                                "tolerance, instead of warning and continuing")

    args = ap.parse_args()
    if args.command == "core":
        cmd_core(args)
    elif args.command == "sweep":
        cmd_sweep(args)
    elif args.command == "style-robust":
        cmd_style_robust(args)
    elif args.command == "multisource":
        cmd_multisource(args)
    elif args.command == "test-sets":
        cmd_test_sets(args)
    elif args.command == "controls":
        cmd_controls(args)
    else:
        cmd_core(args)
        print()
        cmd_sweep(args)


if __name__ == "__main__":
    main()
