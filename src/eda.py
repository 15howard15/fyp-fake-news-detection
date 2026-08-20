
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
from preprocessing import clean_series

EDA_DIR = cfg.RESULTS_DIR / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)


def load_all():
    frames = {}
    for name in ("isot_real", "isot_fake", "liar_real", "liar_fake"):
        path = cfg.PROCESSED_DIR / f"{name}.csv"
        if path.exists():
            frames[name] = pd.read_csv(path)
    if not frames:
        raise FileNotFoundError("Run load_data.py first.")
    return frames


def class_distribution(frames):
    rows = []
    for name, df in frames.items():
        rows.append({"pool": name, "count": len(df)})
    summary = pd.DataFrame(rows)
    summary.to_csv(EDA_DIR / "class_distribution.csv", index=False)

    plt.figure(figsize=(8, 4))
    plt.bar(summary["pool"], summary["count"])
    plt.title("Sample counts per pool")
    plt.ylabel("count")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(EDA_DIR / "class_distribution.png", dpi=150)
    plt.close()
    print(summary.to_string(index=False))


def length_analysis(frames):
    plt.figure(figsize=(9, 5))
    stats = []
    for name, df in frames.items():
        lengths = df["text"].astype(str).str.split().apply(len)
        stats.append({
            "pool": name,
            "mean_words": round(lengths.mean(), 1),
            "median_words": int(lengths.median()),
            "p95_words": int(lengths.quantile(0.95)),
        })
        plt.hist(lengths.clip(upper=600), bins=50, alpha=0.5, label=name)
    plt.title("Word-count distribution (clipped at 600)")
    plt.xlabel("words")
    plt.ylabel("frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(EDA_DIR / "length_distribution.png", dpi=150)
    plt.close()

    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(EDA_DIR / "length_stats.csv", index=False)
    print("\nLength statistics:")
    print(stats_df.to_string(index=False))
    print("\n>> Note the gap between ISOT (long) and LIAR (short). "
          "This IS your domain shift.")


def top_words(frames, n=20):
    from collections import Counter
    out = {}
    for name, df in frames.items():
        cleaned = clean_series(df["text"].head(3000))
        counter = Counter()
        for t in cleaned:
            counter.update(t.split())
        out[name] = counter.most_common(n)
    with open(EDA_DIR / "top_words.txt", "w") as f:
        for name, words in out.items():
            f.write(f"\n=== {name} ===\n")
            for w, c in words:
                f.write(f"{w}: {c}\n")
    print(f"\nTop words written to {EDA_DIR / 'top_words.txt'}")


def main():
    frames = load_all()
    class_distribution(frames)
    length_analysis(frames)
    top_words(frames)
    print(f"\nAll EDA outputs in {EDA_DIR}")


if __name__ == "__main__":
    main()
