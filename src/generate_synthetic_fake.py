
import argparse
import os
import random

import pandas as pd
from tqdm import tqdm

import config as cfg
from gen_common import truncate_article, quality_ok, call_llm

random.seed(cfg.SEED)

STRATEGY_INSTRUCTIONS = {
    "fact_manipulation": (
        "Alter ONE named entity, date, statistic, or numerical value to introduce "
        "a factual inaccuracy (e.g. change '20 dead' to '200 dead')."
    ),
    "context_distortion": (
        "Modify the framing or surrounding context of ONE event to create a "
        "misleading interpretation, WITHOUT changing the core facts (e.g. reframe "
        "a routine policy announcement as a secret backdoor deal)."
    ),
    "tone_adjustment": (
        "Introduce sensational, alarmist, or emotionally charged language around "
        "ONE fact to exaggerate its severity, without inventing new facts."
    ),
    "selective_omission": (
        "Remove ONE critical qualification or contextual detail so the remaining "
        "text becomes one-sided or misleading (e.g. drop that a statistic refers "
        "to a specific subgroup, implying it applies universally)."
    ),
}

SYSTEM_PROMPT = (
    "You are a data-generation tool for academic fake-news-detection research. "
    "You transform a real news article into a synthetic fake variant by applying "
    "exactly ONE targeted change. You MUST keep the rest of the wording almost "
    "identical to the source so the change is subtle and traceable. "
    "Respond ONLY with valid JSON, no markdown, no commentary."
)

USER_TEMPLATE = """Source article:
\"\"\"{article}\"\"\"

Step 1 - Extract 3-6 key facts (entities, numbers, dates, claims) as a list.
Step 2 - Choose ONE fact and apply this transformation: {strategy_desc}
Step 3 - Rewrite the article applying ONLY that change. Keep all other sentences
         as close to the original as possible.

Return JSON with exactly these keys:
{{
  "fact_table": [list of extracted facts as strings],
  "modified_fact": "the single fact you changed, before -> after",
  "synthetic_article": "the full rewritten article text"
}}"""


def generate_one(client, article: str, strategy: str):
    """Call the API once. Returns dict or None on failure."""
    prompt = USER_TEMPLATE.format(
        article=truncate_article(article),
        strategy_desc=STRATEGY_INSTRUCTIONS[strategy],
    )
    return call_llm(client, SYSTEM_PROMPT, prompt, "synthetic_article")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500,
                    help="number of synthetic articles to generate")
    ap.add_argument("--max_retries", type=int, default=2)
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("Set OPENAI_API_KEY first (see README).")

    from openai import OpenAI
    client = OpenAI()

    # IMPORTANT: only draw source articles from the TRAIN split, using the
    # exact same train_test_split call every other script uses (04_build_
    # datasets.py, 04b, 04c, 03b). The previous version used
    # `isot_real.sample(frac=1.0, random_state=SEED)`, a DIFFERENT shuffle
    # algorithm than sklearn's train_test_split with the same seed -- so it
    # did not line up with real_train/real_test at all. Verified: 475/475 of
    # the previously-generated synthetic_fake.csv articles were sourced from
    # real_test (the held-out set), not real_train -- meaning every fake-class
    # training example was a near-duplicate of a real-class TEST example.
    # That's data leakage, and it silently violates the "no train/test
    # overlap" claim in Section 3.2.5 of the thesis. Do not remove this split.
    from sklearn.model_selection import train_test_split
    isot_real = pd.read_csv(cfg.PROCESSED_DIR / "isot_real.csv")
    real_train, _test = train_test_split(
        isot_real, test_size=cfg.TEST_SIZE, random_state=cfg.SEED, shuffle=True
    )
    real = real_train.reset_index(drop=True)

    # Resume support: don't regenerate what we already have.
    out_path = cfg.SYNTHETIC_DIR / "synthetic_fake.csv"
    done = 0
    existing = []
    if out_path.exists():
        existing_df = pd.read_csv(out_path)
        existing = existing_df.to_dict("records")
        done = len(existing_df)
        print(f"Resuming: {done} already generated.")

    results = list(existing)
    pbar = tqdm(total=args.n, initial=done, desc="Generating")
    idx = done
    while len(results) < args.n and idx < len(real):
        article = str(real.iloc[idx]["text"])
        idx += 1
        if len(article.split()) < 30:   # too short to manipulate meaningfully
            continue

        strategy = random.choice(cfg.TRANSFORMATIONS)
        data = None
        for _ in range(args.max_retries):
            data = generate_one(client, article, strategy)
            if data and quality_ok(article, data["synthetic_article"]):
                break
            data = None
        if data is None:
            continue

        results.append({
            "text": data["synthetic_article"],
            "label": cfg.LABEL_FAKE,
            "source": "synthetic",
            "transformation": strategy,
            "modified_fact": data.get("modified_fact", ""),
            "source_text": truncate_article(article, 1000),
        })
        pbar.update(1)

        # checkpoint every 25 so a crash doesn't lose everything
        if len(results) % 25 == 0:
            pd.DataFrame(results).to_csv(out_path, index=False)

    pbar.close()
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nSaved {len(results)} synthetic articles to {out_path}")

    # transformation breakdown for your report
    df = pd.DataFrame(results)
    print("\nTransformation distribution:")
    print(df["transformation"].value_counts().to_string())


if __name__ == "__main__":
    main()
