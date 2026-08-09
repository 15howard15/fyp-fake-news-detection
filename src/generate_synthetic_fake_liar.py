"""
generate_synthetic_fake_liar.py -- Objective 1 fix: generates synthetic
fake news from a SECOND real-news source (LIAR true/mostly-true/half-true
statements), so the training corpus's synthetic fake news is no longer
derived from a single source (ISOT) only. Reuses the exact same four
transformation strategies and the same quality filter as
generate_synthetic_fake.py, so results stay comparable.

LIAR statements are short political claims (median 17 words vs ISOT's much
longer articles), so the minimum source length is lowered to 10 words here
-- these are still single factual claims, just shorter ones than news
articles. Output is tagged source="synthetic_liar" and kept in a separate
file (synthetic_fake_liar.csv) so downstream dataset-builders can choose to
combine it with the ISOT-sourced synthetic_fake.csv or analyse it alone.
"""
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
    "You transform a real statement into a synthetic fake variant by applying "
    "exactly ONE targeted change. You MUST keep the rest of the wording almost "
    "identical to the source so the change is subtle and traceable. "
    "Respond ONLY with valid JSON, no markdown, no commentary."
)

# Same 3-step structure as generate_synthetic_fake.py, worded for short
# statements rather than multi-paragraph articles.
USER_TEMPLATE = """Source statement:
\"\"\"{article}\"\"\"

Step 1 - Extract 1-4 key facts (entities, numbers, dates, claims) as a list.
Step 2 - Choose ONE fact and apply this transformation: {strategy_desc}
Step 3 - Rewrite the statement applying ONLY that change. Keep all other
         wording as close to the original as possible.

Return JSON with exactly these keys:
{{
  "fact_table": [list of extracted facts as strings],
  "modified_fact": "the single fact you changed, before -> after",
  "synthetic_article": "the full rewritten statement text"
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
    ap.add_argument("--n", type=int, default=200,
                    help="number of synthetic statements to generate")
    ap.add_argument("--min_words", type=int, default=10,
                    help="minimum source length in words (LIAR statements are short)")
    ap.add_argument("--max_retries", type=int, default=2)
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("Set OPENAI_API_KEY first (see README).")

    from openai import OpenAI
    client = OpenAI()

    # Same discipline as generate_synthetic_fake.py: only draw source statements
    # from the TRAIN split (same seed/test_size as every other script), even
    # though liar_real is not currently used in any test set. This keeps the
    # door open for future work without silently creating a leakage risk.
    from sklearn.model_selection import train_test_split
    liar_real = pd.read_csv(cfg.PROCESSED_DIR / "liar_real.csv")
    real_train, _test = train_test_split(
        liar_real, test_size=cfg.TEST_SIZE, random_state=cfg.SEED, shuffle=True
    )
    real = real_train.reset_index(drop=True)

    # Resume support: don't regenerate what we already have.
    out_path = cfg.SYNTHETIC_DIR / "synthetic_fake_liar.csv"
    done = 0
    existing = []
    if out_path.exists():
        existing_df = pd.read_csv(out_path)
        existing = existing_df.to_dict("records")
        done = len(existing_df)
        print(f"Resuming: {done} already generated.")

    results = list(existing)
    pbar = tqdm(total=args.n, initial=done, desc="Generating (LIAR source)")
    idx = done
    while len(results) < args.n and idx < len(real):
        statement = str(real.iloc[idx]["text"])
        idx += 1
        if len(statement.split()) < args.min_words:
            continue

        strategy = random.choice(cfg.TRANSFORMATIONS)
        data = None
        for _ in range(args.max_retries):
            data = generate_one(client, statement, strategy)
            if data and quality_ok(statement, data["synthetic_article"], min_words=8):
                break
            data = None
        if data is None:
            continue

        results.append({
            "text": data["synthetic_article"],
            "label": cfg.LABEL_FAKE,
            "source": "synthetic_liar",
            "source_dataset": "liar",
            "transformation": strategy,
            "modified_fact": data.get("modified_fact", ""),
            "source_text": truncate_article(statement, cfg.FULL_SOURCE_CAP),
        })
        pbar.update(1)

        # checkpoint every 25 so a crash doesn't lose everything
        if len(results) % 25 == 0:
            pd.DataFrame(results).to_csv(out_path, index=False)

    pbar.close()
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nSaved {len(results)} synthetic statements to {out_path}")

    if len(real) - idx <= 0 and len(results) < args.n:
        print(f"NOTE: ran out of eligible LIAR-real source statements "
              f"(>= {args.min_words} words) before reaching --n={args.n}. "
              f"Generated {len(results)} instead.")

    # transformation breakdown for your report
    df = pd.DataFrame(results)
    print("\nTransformation distribution:")
    print(df["transformation"].value_counts().to_string())


if __name__ == "__main__":
    main()