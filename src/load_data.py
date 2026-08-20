
import pandas as pd
import config as cfg


def load_isot():
    """ISOT ships as two files: True.csv (real) and Fake.csv (fake)."""
    true_path = cfg.RAW_DIR / "True.csv"
    fake_path = cfg.RAW_DIR / "Fake.csv"

    if not true_path.exists() or not fake_path.exists():
        raise FileNotFoundError(
            f"Put True.csv and Fake.csv in {cfg.RAW_DIR}. See README for the link."
        )

    real = pd.read_csv(true_path)
    fake = pd.read_csv(fake_path)

    real["text"] = (real["title"].fillna("") + ". " + real["text"].fillna("")).str.strip()
    fake["text"] = (fake["title"].fillna("") + ". " + fake["text"].fillna("")).str.strip()

    real = real[["text"]].copy()
    real["label"] = cfg.LABEL_REAL
    real["source"] = "isot"
    real["split"] = "real"

    fake = fake[["text"]].copy()
    fake["label"] = cfg.LABEL_FAKE
    fake["source"] = "isot"
    fake["split"] = "fake"

    return real, fake


def load_liar():
    """
    LIAR is TSV with no header. Column 1 (index 1) is the truth label,
    column 2 (index 2) is the statement. We map 6 labels -> binary.
    """
    cols = [
        "id", "label", "statement", "subject", "speaker", "job",
        "state", "party", "barely_true", "false", "half_true",
        "mostly_true", "pants_fire", "context",
    ]
    frames = []
    for fname in ("train.tsv", "valid.tsv", "test.tsv"):
        path = cfg.RAW_DIR / fname
        if path.exists():
            frames.append(pd.read_csv(path, sep="\t", header=None, names=cols))
    if not frames:
        raise FileNotFoundError(
            f"Put LIAR train/valid/test.tsv in {cfg.RAW_DIR}. See README."
        )

    liar = pd.concat(frames, ignore_index=True)
    liar["binary"] = liar["label"].map(cfg.LIAR_LABEL_MAP)
    liar = liar.dropna(subset=["binary", "statement"])
    liar["binary"] = liar["binary"].astype(int)

    out = pd.DataFrame({
        "text": liar["statement"].astype(str),
        "label": liar["binary"],
        "source": "liar",
    })
    out["split"] = out["label"].map({cfg.LABEL_REAL: "real", cfg.LABEL_FAKE: "fake"})
    real = out[out["label"] == cfg.LABEL_REAL].copy()
    fake = out[out["label"] == cfg.LABEL_FAKE].copy()
    return real, fake


def load_welfake():
    """
    WELFake: 72,134 articles merged from 4 existing fake-news datasets
    (Kaggle, McIntire, Reuters, BuzzFeed). Single CSV: title, text, label.

    Label convention verified directly against the downloaded file, NOT
    trusted from third-party documentation (which claims 0=fake/1=real --
    the opposite of what's actually in the file). Confirmed by matching
    label counts to WELFake's documented 35,028 real / 37,106 fake totals:
    label==0 has 35,028 rows (-> real), label==1 has 37,106 rows (-> fake).
    That happens to already match this project's own LABEL_REAL=0/
    LABEL_FAKE=1 convention, so no inversion is needed -- but map it
    explicitly rather than passing the raw column through silently, so a
    differently-labeled mirror doesn't silently corrupt everything downstream.
    """
    path = cfg.RAW_DIR / "WELFake_Dataset.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Put WELFake_Dataset.csv in {cfg.RAW_DIR}. See README for the link."
        )

    df = pd.read_csv(path)
    df["text"] = (df["title"].fillna("") + ". " + df["text"].fillna("")).str.strip()
    df = df.dropna(subset=["label"])

    WELFAKE_LABEL_MAP = {0: cfg.LABEL_REAL, 1: cfg.LABEL_FAKE}
    df["label"] = df["label"].map(WELFAKE_LABEL_MAP)

    df = df[["text", "label"]].copy()
    df["source"] = "welfake"
    df["split"] = df["label"].map({cfg.LABEL_REAL: "real", cfg.LABEL_FAKE: "fake"})
    real = df[df["label"] == cfg.LABEL_REAL].copy()
    fake = df[df["label"] == cfg.LABEL_FAKE].copy()
    return real, fake


def main():
    print("Loading ISOT...")
    isot_real, isot_fake = load_isot()
    print(f"  ISOT real: {len(isot_real):,} | ISOT fake: {len(isot_fake):,}")

    print("Loading LIAR...")
    liar_real, liar_fake = load_liar()
    print(f"  LIAR real: {len(liar_real):,} | LIAR fake: {len(liar_fake):,}")

    isot_real.to_csv(cfg.PROCESSED_DIR / "isot_real.csv", index=False)
    isot_fake.to_csv(cfg.PROCESSED_DIR / "isot_fake.csv", index=False)
    liar_real.to_csv(cfg.PROCESSED_DIR / "liar_real.csv", index=False)
    liar_fake.to_csv(cfg.PROCESSED_DIR / "liar_fake.csv", index=False)

    welfake_path = cfg.RAW_DIR / "WELFake_Dataset.csv"
    if welfake_path.exists():
        print("Loading WELFake...")
        welfake_real, welfake_fake = load_welfake()
        print(f"  WELFake real: {len(welfake_real):,} | WELFake fake: {len(welfake_fake):,}")
        welfake_real.to_csv(cfg.PROCESSED_DIR / "welfake_real.csv", index=False)
        welfake_fake.to_csv(cfg.PROCESSED_DIR / "welfake_fake.csv", index=False)
    else:
        print(f"  (skipping WELFake -- {welfake_path} not found)")

    print(f"Saved processed files to {cfg.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
