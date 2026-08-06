
"""
generate_counter_style_training.py -- generates the TRAINING-side half of
the Objective 4 fix: paired counter-style twins of articles ALREADY in
train_real_real.csv, so the model sees the same content in both tones with
the SAME true label. This is what generate_style_attack.py's attack set
lacked as a training signal -- that script only ever produced held-out TEST
examples. Mirrors the AdSent / sentiment-agnostic training idea described in
the literature review: decorrelate tone from truth by putting both tones in
BOTH classes during training.

    - real articles (label=real) -> sensationalized twin, STILL label=real
    - fake articles (label=fake) -> neutralized twin,     STILL label=fake

Source articles are the first N rows of train_real_real.csv's real and fake
classes (not the held-out test pool used for the attack set -- no overlap).

Output: data/synthetic/counter_style_training.csv
    columns: text, label, source, attack_type
"""
import argparse
import os

import pandas as pd
from tqdm import tqdm

import config as cfg
from gen_common import (
    truncate_article, quality_ok, call_llm, STYLE_TRANSFER_SYSTEM_PROMPT,
    SENSATIONALIZE_REAL_TEMPLATE, NEUTRALIZE_FAKE_TEMPLATE,
)

# Reuses the exact same prompts as generate_style_attack.py (via gen_common),
# so training and test-time style shifts are drawn from the same distribution.


def generate_one(client, article: str, template: str):
    prompt = template.format(article=truncate_article(article))
    result = call_llm(client, STYLE_TRANSFER_SYSTEM_PROMPT, prompt, "styled_article")
    return result["styled_article"] if result else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_class", type=int, default=100)
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("Set OPENAI_API_KEY first (see README).")

    from openai import OpenAI
    client = OpenAI()

    train_real_real = pd.read_csv(cfg.PROCESSED_DIR / "train_real_real.csv")
    real_pool = train_real_real[train_real_real.label == cfg.LABEL_REAL].reset_index(drop=True)
    fake_pool = train_real_real[train_real_real.label == cfg.LABEL_FAKE].reset_index(drop=True)
    print(f"train_real_real: {len(real_pool)} real / {len(fake_pool)} fake available to pair from")

    out_path = cfg.SYNTHETIC_DIR / "counter_style_training.csv"
    results = []
    done_real, done_fake = 0, 0
    if out_path.exists():
        existing = pd.read_csv(out_path)
        results = existing.to_dict("records")
        done_real = sum(1 for r in results if r["attack_type"] == "sensationalize")
        done_fake = sum(1 for r in results if r["attack_type"] == "neutralize")
        print(f"Resuming: {done_real} real / {done_fake} fake already done.")

    pbar = tqdm(total=args.n_per_class * 2, initial=done_real + done_fake, desc="Counter-style training pairs")

    idx = done_real
    made = done_real
    while made < args.n_per_class and idx < len(real_pool):
        article = str(real_pool.iloc[idx]["text"])
        idx += 1
        if len(article.split()) < 30:
            continue
        styled = generate_one(client, article, SENSATIONALIZE_REAL_TEMPLATE)
        if styled is None or not quality_ok(article, styled, ratio_range=(0.4, 2.5)):
            continue
        results.append({"text": styled, "label": cfg.LABEL_REAL,
                        "source": "counter_style", "attack_type": "sensationalize"})
        made += 1
        pbar.update(1)
        if len(results) % 20 == 0:
            pd.DataFrame(results).to_csv(out_path, index=False)

    idx = done_fake
    made = done_fake
    while made < args.n_per_class and idx < len(fake_pool):
        article = str(fake_pool.iloc[idx]["text"])
        idx += 1
        if len(article.split()) < 30:
            continue
        styled = generate_one(client, article, NEUTRALIZE_FAKE_TEMPLATE)
        if styled is None or not quality_ok(article, styled, ratio_range=(0.4, 2.5)):
            continue
        results.append({"text": styled, "label": cfg.LABEL_FAKE,
                        "source": "counter_style", "attack_type": "neutralize"})
        made += 1
        pbar.update(1)
        if len(results) % 20 == 0:
            pd.DataFrame(results).to_csv(out_path, index=False)

    pbar.close()
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} counter-style training pairs to {out_path}")
    print(df["attack_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
