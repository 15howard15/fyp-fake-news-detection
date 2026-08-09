"""
generate_style_attack.py -- Objective 4: does the model rely on stylistic/
sentiment cues rather than content when judging real vs. fake?

This builds the missing experiment behind Research Objective 4 ("analyze
whether style-diverse synthetic fake news enhances model resistance to
stylistic manipulation and sentiment-based attacks") and the AdSent/
Tahmasebi et al. (2026) framing in the literature review: many detectors
implicitly associate NEUTRAL tone with real news and EMOTIONAL/sensational
tone with fake news. If that's true, you can fool a detector by rewriting
the STYLE of an article while leaving every fact and its true label
unchanged:
    - a REAL article rewritten with sensational/alarmist tone should still
      be predicted real (if the model is robust) or may get misclassified
      as fake (if it's using tone as a shortcut)
    - a FAKE article rewritten with calm, neutral, factual-sounding tone
      should still be predicted fake (if robust) or may get misclassified
      as real (if using tone as a shortcut)

Source articles are drawn from test_indomain.csv (held-out ISOT), so the
attack is applied to data no model has trained on either way.

Output: data/synthetic/style_attack.csv with columns
    text, label, source_text, attack_type
"index" is preserved via a paired "orig_id" so eval_style_robustness.py
can match each attacked article back to its original for a before/after
comparison.
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


def generate_one(client, article: str, template: str):
    prompt = template.format(article=truncate_article(article))
    result = call_llm(client, STYLE_TRANSFER_SYSTEM_PROMPT, prompt, "styled_article")
    return result["styled_article"] if result else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_class", type=int, default=100,
                     help="number of real and number of fake articles to attack (each)")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("Set OPENAI_API_KEY first (see README).")

    from openai import OpenAI
    client = OpenAI()

    test_df = pd.read_csv(cfg.PROCESSED_DIR / "test_indomain.csv")
    real_pool = test_df[test_df.label == cfg.LABEL_REAL].sample(
        frac=1.0, random_state=cfg.SEED).reset_index(drop=True)
    fake_pool = test_df[test_df.label == cfg.LABEL_FAKE].sample(
        frac=1.0, random_state=cfg.SEED).reset_index(drop=True)

    out_path = cfg.SYNTHETIC_DIR / "style_attack.csv"
    done_real, done_fake = 0, 0
    results = []
    if out_path.exists():
        existing = pd.read_csv(out_path)
        results = existing.to_dict("records")
        done_real = sum(1 for r in results if r["attack_type"] == "sensationalize")
        done_fake = sum(1 for r in results if r["attack_type"] == "neutralize")
        print(f"Resuming: {done_real} real / {done_fake} fake already done.")

    pbar = tqdm(total=args.n_per_class * 2, initial=done_real + done_fake, desc="Style attacks")

    # REAL articles -> sensationalize (should still predict real if robust)
    idx = done_real
    made = done_real
    while made < args.n_per_class and idx < len(real_pool):
        article = str(real_pool.iloc[idx]["text"])
        orig_id = f"real_{idx}"
        idx += 1
        if len(article.split()) < 30:
            continue
        styled = generate_one(client, article, SENSATIONALIZE_REAL_TEMPLATE)
        if styled is None or not quality_ok(article, styled, ratio_range=(0.4, 2.5)):
            continue
        results.append({
            "orig_id": orig_id, "text": styled, "label": cfg.LABEL_REAL,
            "source_text": truncate_article(article, cfg.FULL_SOURCE_CAP), "attack_type": "sensationalize",
        })
        made += 1
        pbar.update(1)
        if len(results) % 20 == 0:
            pd.DataFrame(results).to_csv(out_path, index=False)

    # FAKE articles -> neutralize (should still predict fake if robust)
    idx = done_fake
    made = done_fake
    while made < args.n_per_class and idx < len(fake_pool):
        article = str(fake_pool.iloc[idx]["text"])
        orig_id = f"fake_{idx}"
        idx += 1
        if len(article.split()) < 30:
            continue
        styled = generate_one(client, article, NEUTRALIZE_FAKE_TEMPLATE)
        if styled is None or not quality_ok(article, styled, ratio_range=(0.4, 2.5)):
            continue
        results.append({
            "orig_id": orig_id, "text": styled, "label": cfg.LABEL_FAKE,
            "source_text": truncate_article(article, cfg.FULL_SOURCE_CAP), "attack_type": "neutralize",
        })
        made += 1
        pbar.update(1)
        if len(results) % 20 == 0:
            pd.DataFrame(results).to_csv(out_path, index=False)

    pbar.close()
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} style-attacked articles to {out_path}")
    print(df["attack_type"].value_counts().to_string())

    # Also save the matching ORIGINAL (unattacked) subset for paired before/after eval
    orig_rows = []
    for r in results:
        kind, i = r["orig_id"].split("_")
        i = int(i)
        pool = real_pool if kind == "real" else fake_pool
        orig_rows.append({
            "orig_id": r["orig_id"], "text": pool.iloc[i]["text"],
            "label": pool.iloc[i]["label"], "attack_type": r["attack_type"],
        })
    orig_path = cfg.SYNTHETIC_DIR / "style_attack_originals.csv"
    pd.DataFrame(orig_rows).to_csv(orig_path, index=False)
    print(f"Saved matching originals to {orig_path}")


if __name__ == "__main__":
    main()