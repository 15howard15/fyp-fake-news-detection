"""build_datasets.py -- assembles every training composition from the raw pools."""
import argparse
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
