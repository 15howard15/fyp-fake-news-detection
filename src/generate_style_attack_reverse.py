"""
generate_style_attack_reverse.py -- Q4 follow-up: does the style_robust fix
generalize to the OPPOSITE tone shift, or only the direction it was tested
against?

generate_style_attack.py only ever tested one direction per class:
    - REAL article -> sensationalized (should still predict real)
    - FAKE article -> neutralized (should still predict fake)
That's the direction that matches the intuitive "fake news sounds dramatic"
shortcut, so it's also the easiest direction to defend against by design.
This script builds the REVERSE pairing to check the fix isn't narrower than
it looks:
    - REAL article -> neutralized (should still predict real -- real articles
      are often already fairly neutral in tone, so this direction may move
      the needle less; that's an informative result either way, not a failure
      of the attack)
    - FAKE article -> sensationalized (should still predict fake)

Source articles are drawn from the SAME held-out pools as generate_style_attack.py (test_indomain,
split by cfg.SEED into real_pool/fake_pool), but explicitly EXCLUDING every
orig_id already used in style_attack_originals.csv, so the two attack sets
never share an article -- verified via IDs read back from that file, not an
assumed offset.

Output: data/synthetic/style_attack_reverse.csv (columns match generate_
style_attack.py's output: text, label, source_text, attack_type) and the
matching style_attack_reverse_originals.csv, in the same orig_id/text/label/
attack_type shape eval_style_robustness.py-style scripts expect, but attack_type here
is "neutralize_real" / "sensationalize_fake" to keep the two attack sets
unambiguous in any combined analysis.
"""
import argparse
import os

import pandas as pd
from tqdm import tqdm

import config as cfg
from gen_common import truncate_article, quality_ok, call_llm, STYLE_TRANSFER_SYSTEM_PROMPT

NEUTRALIZE_REAL_TEMPLATE = """Source article (this is REAL, factual news):
\"\"\"{article}\"\"\"

Rewrite this article using calm, neutral, measured, factual-sounding
journalistic tone -- plain wire-service style, no embellishment. Do NOT
change, add, or remove any fact, name, date, or number. The rewritten
article must remain exactly as true as the original.

Return JSON with exactly these keys:
{{
  "styled_article": "the full rewritten article text"
}}"""

SENSATIONALIZE_FAKE_TEMPLATE = """Source article (this is FAKE / false news):
\"\"\"{article}\"\"\"

Rewrite this article using sensational, alarmist, emotionally charged
language -- as if a tabloid or clickbait outlet were reporting the SAME
false story. Exaggerate tone and framing only. Do NOT change, add, or
remove any fact, name, date, or number, and do NOT correct the false
claim -- the rewritten article must remain exactly as FALSE as the
original, just more dramatic in tone.

Return JSON with exactly these keys:
{{
  "styled_article": "the full rewritten article text"
}}"""


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

    # Exclude every article the ORIGINAL (forward-direction) attack already
    # used, so the two attack sets never overlap -- read back from disk
    # rather than assumed from an index offset, since quality-filter retries
    # mean the original script's actual stopping index isn't fixed.
    used_real_idx, used_fake_idx = set(), set()
    orig_path = cfg.SYNTHETIC_DIR / "style_attack_originals.csv"
    if orig_path.exists():
        prior = pd.read_csv(orig_path)
        for oid in prior["orig_id"]:
            kind, i = oid.split("_")
            (used_real_idx if kind == "real" else used_fake_idx).add(int(i))
        print(f"Excluding {len(used_real_idx)} real / {len(used_fake_idx)} fake "
              f"articles already used by the forward-direction attack.")

    out_path = cfg.SYNTHETIC_DIR / "style_attack_reverse.csv"
    done_real, done_fake = 0, 0
    results = []
    if out_path.exists():
        existing = pd.read_csv(out_path)
        results = existing.to_dict("records")
        done_real = sum(1 for r in results if r["attack_type"] == "neutralize_real")
        done_fake = sum(1 for r in results if r["attack_type"] == "sensationalize_fake")
        print(f"Resuming: {done_real} real / {done_fake} fake already done.")

    pbar = tqdm(total=args.n_per_class * 2, initial=done_real + done_fake, desc="Reverse style attacks")

    # REAL articles -> neutralize (should still predict real if robust)
    idx = 0
    made = done_real
    done_ids = {r["orig_id"] for r in results}
    while made < args.n_per_class and idx < len(real_pool):
        orig_id = f"real_{idx}"
        if idx in used_real_idx or orig_id in done_ids:
            idx += 1
            continue
        article = str(real_pool.iloc[idx]["text"])
        idx += 1
        if len(article.split()) < 30:
            continue
        styled = generate_one(client, article, NEUTRALIZE_REAL_TEMPLATE)
        if styled is None or not quality_ok(article, styled, ratio_range=(0.4, 2.5)):
            continue
        results.append({
            "orig_id": orig_id, "text": styled, "label": cfg.LABEL_REAL,
            "source_text": truncate_article(article, cfg.FULL_SOURCE_CAP), "attack_type": "neutralize_real",
        })
        made += 1
        pbar.update(1)
        if len(results) % 20 == 0:
            pd.DataFrame(results).to_csv(out_path, index=False)

    # FAKE articles -> sensationalize (should still predict fake if robust)
    idx = 0
    made = done_fake
    while made < args.n_per_class and idx < len(fake_pool):
        orig_id = f"fake_{idx}"
        if idx in used_fake_idx or orig_id in done_ids:
            idx += 1
            continue
        article = str(fake_pool.iloc[idx]["text"])
        idx += 1
        if len(article.split()) < 30:
            continue
        styled = generate_one(client, article, SENSATIONALIZE_FAKE_TEMPLATE)
        if styled is None or not quality_ok(article, styled, ratio_range=(0.4, 2.5)):
            continue
        results.append({
            "orig_id": orig_id, "text": styled, "label": cfg.LABEL_FAKE,
            "source_text": truncate_article(article, cfg.FULL_SOURCE_CAP), "attack_type": "sensationalize_fake",
        })
        made += 1
        pbar.update(1)
        if len(results) % 20 == 0:
            pd.DataFrame(results).to_csv(out_path, index=False)

    pbar.close()
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} reverse style-attacked articles to {out_path}")
    print(df["attack_type"].value_counts().to_string())

    # Matching ORIGINAL (unattacked) subset for paired before/after eval
    orig_rows = []
    for r in results:
        kind, i = r["orig_id"].split("_")
        i = int(i)
        pool = real_pool if kind == "real" else fake_pool
        orig_rows.append({
            "orig_id": r["orig_id"], "text": pool.iloc[i]["text"],
            "label": pool.iloc[i]["label"], "attack_type": r["attack_type"],
        })
    rev_orig_path = cfg.SYNTHETIC_DIR / "style_attack_reverse_originals.csv"
    pd.DataFrame(orig_rows).to_csv(rev_orig_path, index=False)
    print(f"Saved matching originals to {rev_orig_path}")


if __name__ == "__main__":
    main()