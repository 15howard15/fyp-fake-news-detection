"""Generate synthetic fake news by altering one fact in a real ISOT article."""

import argparse
import os
import random

import pandas as pd
from tqdm import tqdm

import config as cfg
from gen_common import (truncate_article, quality_ok, call_llm,
                        SYMMETRIC_PARAPHRASE_INSTRUCTION, SYMMETRIC_SYSTEM_BASE)

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

LENGTH_SPECS = {
    "short":  (25,  "a single short social-media-style snippet of about 25 words -- "
                    "one or two sentences, no headline, no byline"),
    "medium": (100, "a condensed news summary of about 100 words -- roughly one "
                    "short paragraph, as a news aggregator would show it"),
    "long":   (400, "a full-length news article of about 400 words, written in "
                    "normal newswire style"),
}

LENGTH_TEMPLATE = """Source article:
\"\"\"{article}\"\"\"

Step 1 - Extract 3-6 key facts (entities, numbers, dates, claims) as a list.
Step 2 - Choose ONE fact and apply this transformation: {strategy_desc}
Step 3 - Write {length_desc}, reporting this story WITH that single altered
         fact in it. Aim for roughly {target} words. Every other fact you
         include must stay exactly as it is in the source -- change one thing
         and one thing only. The altered fact MUST appear in what you write.

Return JSON with exactly these keys:
{{
  "fact_table": [list of extracted facts as strings],
  "modified_fact": "the single fact you changed, before -> after",
  "synthetic_article": "the text you wrote"
}}"""


SYSTEM_PROMPT = (
    "You are a data-generation tool for academic fake-news-detection research. "
    "You transform a real news article into a synthetic fake variant by applying "
    "exactly ONE targeted change. You MUST keep the rest of the wording almost "
    "identical to the source so the change is subtle and traceable. "
    "Respond ONLY with valid JSON, no markdown, no commentary."
)

SYMMETRIC_SYSTEM_PROMPT = (
    SYMMETRIC_SYSTEM_BASE +
    " In addition to rewriting, you introduce exactly ONE targeted factual "
    "change. Every other detail must stay faithful to the source."
)

SYMMETRIC_TEMPLATE = """Source article:
\"\"\"{article}\"\"\"

Step 1 - Extract 3-6 key facts (entities, numbers, dates, claims) as a list.
Step 2 - Choose ONE fact and apply this transformation: {strategy_desc}
Step 3 - {paraphrase}

Your rewrite must be about {target} words long -- the same length as the source.
Do not summarise or condense: cover every point the source covers.

Apply the change from Step 2 as you rewrite. Every fact you did NOT choose must
survive the rewrite unchanged in meaning. The altered fact MUST appear in what
you write.

Return JSON with exactly these keys:
{{
  "fact_table": [list of extracted facts as strings],
  "modified_fact": "the single fact you changed, before -> after",
  "modified_fact_as_written": "copy the ONE sentence from your rewritten article that carries the altered fact, word for word",
  "synthetic_article": "the full rewritten article text"
}}"""


LENGTH_SYSTEM_PROMPT = (
    "You are a data-generation tool for academic fake-news-detection research. "
    "You retell a real news article at a requested length, introducing exactly "
    "ONE targeted factual change. Every other detail you include must remain "
    "faithful to the source. Hit the requested length as closely as you can. "
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


def generate_one(client, article: str, strategy: str, length: str = None,
                 symmetric: bool = False):
    """Call the API once."""
    if symmetric:
        window = truncate_article(article)
        prompt = SYMMETRIC_TEMPLATE.format(
            article=window,
            strategy_desc=STRATEGY_INSTRUCTIONS[strategy],
            paraphrase=SYMMETRIC_PARAPHRASE_INSTRUCTION,
            target=len(window.split()),
        )
        return call_llm(client, SYMMETRIC_SYSTEM_PROMPT, prompt, "synthetic_article")

    if length is None:
        prompt = USER_TEMPLATE.format(
            article=truncate_article(article),
            strategy_desc=STRATEGY_INSTRUCTIONS[strategy],
        )
    else:
        target, desc = LENGTH_SPECS[length]
        prompt = LENGTH_TEMPLATE.format(
            article=truncate_article(article),
            strategy_desc=STRATEGY_INSTRUCTIONS[strategy],
            length_desc=desc, target=target,
        )
    system = SYSTEM_PROMPT if length is None else LENGTH_SYSTEM_PROMPT
    return call_llm(client, system, prompt, "synthetic_article")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500,
                    help="number of synthetic articles to generate")
    ap.add_argument("--max_retries", type=int, default=2)
    ap.add_argument("--lengths", nargs="+", choices=list(LENGTH_SPECS), default=None,
                    help="generate length-controlled fakes instead of source-length "
                         "ones, splitting --n evenly across the named buckets "
                         "(e.g. --lengths short medium long). Writes to a separate "
                         "file so synthetic_fake.csv is never overwritten.")
    ap.add_argument("--symmetric", action="store_true",
                    help="rewrite the whole article to the SAME depth as "
                         "generate_synthetic_real.py --symmetric while altering one "
                         "fact, so the two classes differ only in whether a fact "
                         "changed. Removes the authorship shortcut where the fake "
                         "class sits closer to the source wording (measured: 65.9%% "
                         "retained vs 44.0%% for synthetic-real). Writes to a "
                         "separate file; synthetic_fake.csv is never overwritten.")
    ap.add_argument("--out", default=None,
                    help="output filename under data/synthetic/ (default: "
                         "synthetic_fake.csv, or synthetic_fake_mixedlen.csv "
                         "when --lengths is used, or synthetic_fake_sym.csv "
                         "when --symmetric is used)")
    args = ap.parse_args()

    if args.symmetric and args.lengths:
        ap.error("--symmetric and --lengths change the same prompt in different "
                 "directions; run them separately into separate files.")

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("Set OPENAI_API_KEY first (see README).")

    from openai import OpenAI
    client = OpenAI()

    from sklearn.model_selection import train_test_split
    isot_real = pd.read_csv(cfg.PROCESSED_DIR / "isot_real.csv")
    real_train, _test = train_test_split(
        isot_real, test_size=cfg.TEST_SIZE, random_state=cfg.SEED, shuffle=True
    )
    real = real_train.reset_index(drop=True)

    if args.lengths:
        buckets = list(args.lengths)
        per = args.n // len(buckets)
        plan = []
        for i, b in enumerate(buckets):
            k = per + (args.n - per * len(buckets) if i == len(buckets) - 1 else 0)
            plan += [b] * k
    else:
        plan = [None] * args.n

    default_name = ("synthetic_fake_mixedlen.csv" if args.lengths
                    else "synthetic_fake_sym.csv" if args.symmetric
                    else "synthetic_fake.csv")
    out_path = cfg.SYNTHETIC_DIR / (args.out or default_name)
    done = 0
    existing = []
    if out_path.exists():
        existing_df = pd.read_csv(out_path)
        existing = existing_df.to_dict("records")
        done = len(existing_df)
        print(f"Resuming: {done} already generated.")

    if args.lengths:
        region = len(real) // len(buckets)
        cursors = {b: i * region for i, b in enumerate(buckets)}
        limits = {b: (i + 1) * region if i < len(buckets) - 1 else len(real)
                  for i, b in enumerate(buckets)}
        for row in existing:
            b = row.get("length")
            if b in cursors:
                cursors[b] += 1
    else:
        cursors, limits = {None: done}, {None: len(real)}

    results = list(existing)
    pbar = tqdm(total=args.n, initial=done, desc="Generating")
    while len(results) < len(plan):
        bucket = plan[len(results)]
        if cursors[bucket] >= limits[bucket]:
            print(f"\n(source pool exhausted for bucket {bucket!r} -- "
                  f"dropping its remaining slots)")
            plan = plan[:len(results)] + [p for p in plan[len(results):] if p != bucket]
            continue
        article = str(real.iloc[cursors[bucket]]["text"])
        cursors[bucket] += 1
        if len(article.split()) < 30:
            continue

        target = LENGTH_SPECS[bucket][0] if bucket else None
        if args.symmetric:
            target = len(truncate_article(article).split())
        strategy = random.choice(cfg.TRANSFORMATIONS)
        data = None
        for _ in range(args.max_retries):
            data = generate_one(client, article, strategy, bucket,
                                symmetric=args.symmetric)
            if data and quality_ok(article, data["synthetic_article"],
                                   target_words=target,
                                   target_tol=(0.85, 1.6) if args.symmetric else 0.6):
                break
            data = None
        if data is None:
            continue

        row = {
            "text": data["synthetic_article"],
            "label": cfg.LABEL_FAKE,
            "source": "synthetic",
            "transformation": strategy,
            "modified_fact": data.get("modified_fact", ""),
            "source_text": truncate_article(article, cfg.FULL_SOURCE_CAP),
        }
        if bucket:
            row["length"] = bucket
        if args.symmetric:
            row["modified_fact_as_written"] = data.get("modified_fact_as_written", "")
        results.append(row)
        pbar.update(1)

        if len(results) % 25 == 0:
            pd.DataFrame(results).to_csv(out_path, index=False)

    pbar.close()
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nSaved {len(results)} synthetic articles to {out_path}")

    df = pd.DataFrame(results)
    print("\nTransformation distribution:")
    print(df["transformation"].value_counts().to_string())
    if "length" in df.columns:
        w = df["text"].astype(str).str.split().str.len()
        print("\nLength bucket   n   target   median words")
        for b in df["length"].dropna().unique():
            sel = w[df["length"] == b]
            print(f"  {b:10s} {len(sel):4d}   {LENGTH_SPECS[b][0]:5d}   {int(sel.median()):6d}")


if __name__ == "__main__":
    main()
