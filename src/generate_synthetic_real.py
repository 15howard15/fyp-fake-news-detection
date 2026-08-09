"""
generate_synthetic_real.py -- generate SYNTHETIC REAL news (paraphrase-only,
fact-preserving rewrites of authentic ISOT real articles).

WHY THIS EXISTS
----------------
Your fake-class synthetic data (generate_synthetic_fake.py) is LLM-generated, but
your real-class training data is always human-written (ISOT). That means a
classifier trained on real_real / mixed / real_syn could, in principle, be
learning "is this text LLM-authored?" instead of "is this text fake?" -- the
same shortcut-learning risk your Chapter 2 lit review raises (Section 3.3.3)
but never actually tests.

This script generates the control condition: real news, paraphrased by the
SAME LLM family, with every fact/name/date/number kept identical. Pairing this
with real fake vs. synthetic fake gives you two new training compositions:

    C2 = synthetic-real + real-fake       (isolates: does swapping the REAL
                                            side to LLM-authored text alone
                                            change performance?)
    C3 = synthetic-real + synthetic-fake  (both classes LLM-authored -- if a
                                            model can still discriminate here,
                                            it's using content, not authorship)

USAGE
-----
    python src/generate_synthetic_real.py --n 1000

Costs API calls just like generate_synthetic_fake.py -- run it yourself with
your own OPENAI_API_KEY. Supports the same resume/checkpoint behaviour.
After it finishes, run build_synthetic_real_datasets.py to assemble C2/C3.
"""
import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import config as cfg
from gen_common import truncate_article, quality_ok, call_llm

SYSTEM_PROMPT = (
    "You are a data-generation tool for academic fake-news-detection research. "
    "You rewrite a real news article in different wording while preserving it "
    "as 100% factually identical: every name, date, number, location, quote, "
    "and claim must remain exactly as given. Do NOT add, remove, exaggerate, "
    "or distort any fact -- this is a pure style/wording paraphrase, not a "
    "fake-news transformation. Respond ONLY with valid JSON, no markdown."
)

USER_TEMPLATE = """Source article:
\"\"\"{article}\"\"\"

Rewrite this article in fresh wording (different sentence structure, different
phrasing, different word choices) while keeping every fact, name, date,
number, and claim exactly as stated in the source. The result should read as
independently-written but must remain fully factually equivalent.

Return JSON with exactly this key:
{{
  "paraphrased_article": "the full rewritten article text"
}}"""


def generate_one(client, article: str):
    prompt = USER_TEMPLATE.format(article=truncate_article(article))
    return call_llm(client, SYSTEM_PROMPT, prompt, "paraphrased_article")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000,
                     help="number of synthetic-real articles to generate")
    ap.add_argument("--max_retries", type=int, default=2)
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("Set OPENAI_API_KEY first (see README).")

    from openai import OpenAI
    client = OpenAI()

    isot_real = pd.read_csv(cfg.PROCESSED_DIR / "isot_real.csv")

    # IMPORTANT: only draw from the TRAIN split, using the exact same
    # train_test_split call as build_core_datasets.py / build_augmented_datasets.py
    # / build_test_sets.py. If we instead
    # re-shuffled isot_real independently here, we could accidentally
    # paraphrase articles that are held out in real_test / test_indomain /
    # test_crossdomain -- leaking their semantic content into training even
    # though the exact text differs.
    real_train, _test = train_test_split(
        isot_real, test_size=cfg.TEST_SIZE, random_state=cfg.SEED, shuffle=True
    )
    real_train = real_train.reset_index(drop=True)

    out_path = cfg.SYNTHETIC_DIR / "synthetic_real.csv"
    done = 0
    existing = []
    if out_path.exists():
        existing_df = pd.read_csv(out_path)
        existing = existing_df.to_dict("records")
        done = len(existing_df)
        print(f"Resuming: {done} already generated.")

    results = list(existing)
    pbar = tqdm(total=args.n, initial=done, desc="Generating synthetic-real")
    idx = done
    while len(results) < args.n and idx < len(real_train):
        article = str(real_train.iloc[idx]["text"])
        idx += 1
        if len(article.split()) < 30:
            continue

        data = None
        for _ in range(args.max_retries):
            data = generate_one(client, article)
            if data and quality_ok(article, data["paraphrased_article"]):
                break
            data = None
        if data is None:
            continue

        results.append({
            "text": data["paraphrased_article"],
            "label": cfg.LABEL_REAL,
            "source": "synthetic_real",
            "source_text": truncate_article(article, cfg.FULL_SOURCE_CAP),
        })
        pbar.update(1)

        if len(results) % 25 == 0:
            pd.DataFrame(results).to_csv(out_path, index=False)

    pbar.close()
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nSaved {len(results)} synthetic-real articles to {out_path}")


if __name__ == "__main__":
    main()