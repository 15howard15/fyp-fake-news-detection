
import re

import pandas as pd
from sklearn.model_selection import train_test_split

import config as cfg
from generate_synthetic_fake import LENGTH_SPECS


def load(name):
    return pd.read_csv(cfg.PROCESSED_DIR / f"{name}.csv")


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

    # ---- real_syn_mixedlen: the same recipe with the length confound removed ----
    #
    # This is an ADDITIONAL composition, not a replacement for real_syn. The two
    # differ in exactly one variable -- the length distribution of both classes --
    # so the gap between them measures how much of real_syn's cross-domain
    # collapse was ever about length. Swapping it in would destroy that
    # comparison and every result already derived from real_syn.
    #
    # Two properties are deliberately preserved from real_syn so length really is
    # the only thing that moved:
    #
    #   1. The pairing. In real_syn, 499 of the 500 real rows are the very
    #      articles the synthetic fakes were generated from -- the set is 500
    #      minimal pairs, article X labelled real against X-with-one-fact-changed
    #      labelled fake. That is a strong property (the only systematic
    #      difference within a pair is the altered fact) and it is kept here.
    #   2. The class balance and count.
    #
    # What changes: each pair now lives at ~25, ~100 or ~400 words instead of
    # both sides sitting at the source article's full length. Because BOTH sides
    # of every pair are cut to the same target, length carries no information
    # about the label at all -- where naively swapping in mixed-length fakes
    # against full-length reals would have made "short => fake" a free win on
    # two thirds of the fake class.
    ml_path = cfg.SYNTHETIC_DIR / "synthetic_fake_mixedlen.csv"
    if not ml_path.exists():
        print(f"\n(skipping real_syn_mixedlen -- {ml_path.name} not found; "
              f"run generate_synthetic_fake.py --lengths short medium long)")
    else:
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
        ml_df = (pd.DataFrame(ml_rows)
                 .sample(frac=1.0, random_state=cfg.SEED).reset_index(drop=True))
        ml_df.to_csv(cfg.PROCESSED_DIR / "train_real_syn_mixedlen.csv", index=False)
        w = ml_df["text"].str.split().str.len()
        print(f"\n  train_real_syn_mixedlen: {len(ml_df):,} "
              f"({(ml_df.label==0).sum()} real / {(ml_df.label==1).sum()} fake)")
        # The whole point of the composition is that these two medians match.
        # Print them so a mismatch is visible at build time rather than being
        # discovered later in the results.
        print(f"    median words -- real {int(w[ml_df.label==0].median())} / "
              f"fake {int(w[ml_df.label==1].median())}")

    print("\nAll training sets + shared test set built.")


if __name__ == "__main__":
    main()
