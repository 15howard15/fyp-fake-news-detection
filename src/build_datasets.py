"""
build_datasets.py -- assembles every training composition from the raw pools.

Merged from build_core_datasets.py, build_swap_sweep_datasets.py,
build_style_robust_dataset.py and build_multisource_dataset.py.

The merge is not only about file count. Three of those scripts independently
re-derived the ISOT train/test split with their own copy of the same
train_test_split call, and the sweep's validity depends on every composition
drawing from the EXACT same 500 real rows -- build_swap_sweep_datasets.py's
docstring asserted that as a convention maintained by copy-paste. isot_pools()
below makes it structural: there is now one split, and every subcommand takes
its pools from it, so the compositions cannot silently drift apart.

Subcommands, in pipeline order:

  core          real_real / mixed / real_syn, the shared test set, and (if the
                mixed-length corpus exists) real_syn_mixedlen
  sweep         swap_025 / swap_075, the two synthetic-fraction points that
                aren't already covered by core's three
  style-robust  train_real_real + counter-style twins  (run after
                `generate_style.py counter-training`)
  multisource   real_syn with half its fake class re-sourced from LIAR
                (run after `generate_synthetic_fake_liar.py`)

Usage:
    python src/build_datasets.py core
    python src/build_datasets.py sweep
    python src/build_datasets.py style-robust
    python src/build_datasets.py multisource
    python src/build_datasets.py all        # core -> sweep, the two that pair
"""
import argparse
import re

import pandas as pd
from sklearn.model_selection import train_test_split

import config as cfg
from generate_synthetic_fake import LENGTH_SPECS

FRACTIONS = [0.25, 0.75]  # 0.00/0.50/1.00 already exist as real_real/mixed/real_syn


def isot_pools():
    """The one canonical ISOT split, shared by every composition.

    Every caller needs the same 80/20 real split and the same shuffled fake
    pool; if two callers derived these separately and one changed, the
    compositions would stop being comparable while still looking fine. The
    fake pool is deliberately NOT reset_index'd -- .head(n) slices
    positionally either way, and leaving it matches the ordering the existing
    datasets were built with.
    """
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
    """Cut a real article down to about `target` words, keeping whole sentences.

    Two constraints, and a naive implementation cannot satisfy both:

    - Sentences must stay intact. Word-count truncation leaves the real class as
      mid-sentence fragments while the synthetic fakes are complete, well-formed
      snippets, which hands the classifier "is this a fragment?" as a brand-new
      shortcut -- the opposite of what a length control is for.
    - The result must actually land near the target. Taking sentences from the
      start until the target is reached systematically OVERSHOOTS at short
      targets, because newswire ledes are long: the first sentence alone is
      routinely 45+ words against a 25-word target. That leaves real text
      reliably longer than fake text in the short bucket and rebuilds the
      length shortcut this composition exists to remove.

    So: pick the run of consecutive sentences whose total length is closest to
    the target, rather than always starting at sentence one. Any window of an
    ISOT article is still genuine unaltered ISOT text, so nothing about the
    label changes -- only which part of the article is kept.
    """
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
                break        # longer windows from i only get worse
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

    # ---- Shared test set: held-out real + ALL liar fake ----
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
    # Keep fake count = min(real_train, available isot fake not in test, synthetic).
    # IMPORTANT: this must be the SAME target count for all three compositions,
    # otherwise real_real/mixed vs real_syn aren't comparable (real_syn is 100%
    # synthetic, so it needs `len(synthetic)` fake examples available, not
    # `len(synthetic) * 2` — that x2 only makes sense for the 50/50 "mixed" case
    # and was silently giving real_syn half the fake count of the other two).
    n_fake = min(n_real, len(isot_fake_pool), len(synthetic))
    print(f"Using {n_fake:,} fake samples per composition "
          f"(same count for real_real / mixed / real_syn — capped by synthetic "
          f"supply: {len(synthetic):,} available).")

    # Cap the REAL side to n_fake too (balanced 1:1), instead of using the
    # full real_train (~17k). Uncapped, real_real/mixed/real_syn end up at a
    # ~34:1 real:fake ratio while the C2/C3 synthetic-real controls (built in
    # build_synthetic_real_datasets.py) are 1:1 -- an uncontrolled second
    # variable (class balance) riding alongside the one you actually want to
    # isolate (fake-class source). Confusion matrices showed LR/SVM at 34:1
    # collapse to predicting ~100% "real" regardless of what the fake class
    # contains, which was masquerading as "synthetic fake breaks
    # generalization." Capping here makes real_real/mixed/real_syn/C2/C3 all
    # comparable at the same ratio.
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
    """real_syn_mixedlen: the same recipe with the length confound removed.

    This is an ADDITIONAL composition, not a replacement for real_syn. The two
    differ in exactly one variable -- the length distribution of both classes --
    so the gap between them measures how much of real_syn's cross-domain
    collapse was ever about length. Swapping it in would destroy that
    comparison and every result already derived from real_syn.

    Two properties are deliberately preserved from real_syn so length really is
    the only thing that moved:

      1. The pairing. In real_syn, 499 of the 500 real rows are the very
         articles the synthetic fakes were generated from -- the set is 500
         minimal pairs, article X labelled real against X-with-one-fact-changed
         labelled fake. That is a strong property (the only systematic
         difference within a pair is the altered fact) and it is kept here.
      2. The class balance and count.

    What changes: each pair now lives at ~25, ~100 or ~400 words instead of
    both sides sitting at the source article's full length. Because BOTH sides
    of every pair are cut to the same target, length carries no information
    about the label at all -- where naively swapping in mixed-length fakes
    against full-length reals would have made "short => fake" a free win on
    two thirds of the fake class.
    """
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
    # The whole point of the composition is that these two medians match.
    # Print them so a mismatch is visible at build time rather than being
    # discovered later in the results.
    print(f"    median words -- real {int(w[ml_df.label==0].median())} / "
          f"fake {int(w[ml_df.label==1].median())}")


def cmd_sweep(args):
    """The balance-CONTROLLED synthetic-fraction sweep.

    This replaced an earlier "augmentation" design that ADDED synthetic fake on
    top of a fixed real-fake baseline, growing the fake class without growing
    the real class -- 500 real vs 1,000 fake, a 1:2 imbalance. Confusion
    matrices showed LR/SVM/BERT collapsing to ~100% recall / ~57% precision
    cross-domain under that design, almost exactly what a trivial "always
    predict fake" classifier would score: the same majority-class-collapse
    mechanism already fixed on the replacement axis, re-introduced in the
    opposite direction. That design was dropped and its scripts removed.

    Here the TOTAL fake count stays fixed at 500 (matching the 500 real rows)
    and only the synthetic FRACTION of it varies, which isolates "does
    synthetic content help" from "does adding synthetic unbalance the classes":

        swap_000 (  0% synthetic) = train_real_real   (built by `core`)
        swap_025 ( 25% synthetic) = 375 real-fake + 125 synthetic-fake
        swap_050 ( 50% synthetic) = train_mixed       (built by `core`)
        swap_075 ( 75% synthetic) = 125 real-fake + 375 synthetic-fake
        swap_100 (100% synthetic) = train_real_syn    (built by `core`)

    Only 025 and 075 are built here; the other three are re-used by name in
    run_swap_sweep_experiment.py. All five draw from the same isot_pools(),
    so they differ by ONLY the real-fake/synthetic-fake split.
    """
    synthetic = pd.read_csv(cfg.SYNTHETIC_DIR / "synthetic_fake.csv")[
        ["text", "label", "source"]]
    real_train, _test, isot_fake_pool = isot_pools()

    n_fake = min(len(real_train), len(isot_fake_pool), len(synthetic))  # 500
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
    """train_style_robust: the actual test of Objective 4's hypothesis.

    Unlike `mixed` (which adds generic synthetic fake news, not built for
    style-robustness), this adds PAIRED counter-style twins of articles already
    in train_real_real -- the same content in both tones, same true label -- so
    the model can no longer use tone as a shortcut between the classes.

        train_style_robust = train_real_real (500 real / 500 fake)
                           + counter_style_training.csv (100 sensationalized-
                             real + 100 neutralized-fake, twins of articles
                             already in train_real_real)
                           = 600 real / 600 fake -- still balanced, so this
                             isn't confounded by the imbalance issue found
                             earlier in the project.

    Run AFTER `generate_style.py counter-training`.
    """
    cols = ["text", "label", "source"]
    real_real = pd.read_csv(cfg.PROCESSED_DIR / "train_real_real.csv")[cols]
    counter = pd.read_csv(cfg.SYNTHETIC_DIR / "counter_style_training.csv")[cols]
    df = _shuffled(pd.concat([real_real, counter], ignore_index=True))
    _write(df, "train_style_robust")


def cmd_multisource(args):
    """real_syn with half its synthetic fake class re-sourced from LIAR.

    Everything else -- the real class, and the TOTAL fake count -- is kept
    identical to train_real_syn.csv, so this is a direct, single-variable
    ablation: same size, same real class, only the source mix of the synthetic
    fake class changes. Compare against train_real_syn.csv to test whether
    "diverse sources" (Objective 1) actually changes detection performance.
    """
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
            f"{len(isot_synthetic)} exist in {isot_syn_path}.")

    fake_part = pd.concat([isot_synthetic.head(n_isot),
                           liar_synthetic.head(n_liar)], ignore_index=True)
    df = _shuffled(pd.concat([real_part, fake_part], ignore_index=True))
    _write(df, "train_real_syn_multisource")
    print(f"      ({n_isot} ISOT-sourced synthetic + {n_liar} LIAR-sourced)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="command", required=True)
    for name, helptext in [
        ("core", "real_real / mixed / real_syn + shared test set (+ mixedlen)"),
        ("sweep", "swap_025 / swap_075, the synthetic-fraction sweep points"),
        ("style-robust", "train_real_real + counter-style twins"),
        ("multisource", "real_syn with half the fake class re-sourced from LIAR"),
        ("all", "core then sweep, the two that must stay in step"),
    ]:
        sub.add_parser(name, help=helptext)

    args = ap.parse_args()
    if args.command == "core":
        cmd_core(args)
    elif args.command == "sweep":
        cmd_sweep(args)
    elif args.command == "style-robust":
        cmd_style_robust(args)
    elif args.command == "multisource":
        cmd_multisource(args)
    else:
        cmd_core(args)
        print()
        cmd_sweep(args)


if __name__ == "__main__":
    main()
