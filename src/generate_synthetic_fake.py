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

# ----------------------------------------------------------------------
# Length control (--lengths)
# ----------------------------------------------------------------------
# The default generator preserves the source article's length, because it
# applies one edit and leaves the rest of the wording alone. That is the right
# design for "is this fact false?", but it hands the classifier a shortcut:
# synthetic fakes have a median of 376 words against 369 for ISOT real, while
# LIAR statements have a median of 16. A detector can therefore separate the
# training classes on length-correlated cues and still look like it learned
# something about truth -- and the RQ3 length sweep already showed this project
# is exposed to exactly that confound.
#
# Generating the same fact-manipulation at three very different lengths breaks
# the correlation between length and label without changing what makes the text
# false, so a model trained on the mixed-length corpus cannot use length as a
# proxy for the label.
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

# --symmetric: same paraphrase depth as generate_synthetic_real.py, so the two
# classes differ only in whether a fact was altered. See the note in
# gen_common.py for the measured asymmetry this exists to remove.
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


# The default system prompt demands the rest of the wording stay near-identical
# to the source, which is unsatisfiable when the target is 25 words from a
# 400-word article. Length mode keeps the part that matters -- exactly one
# altered fact, everything else faithful -- and drops the verbatim-wording
# requirement, which is a means to that end rather than the end itself.
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
    """Call the API once. Returns dict or None on failure.

    length=None and symmetric=False keep the original behaviour exactly (same
    prompt, same system message), so existing synthetic_fake.csv rows stay
    reproducible from this file. The two modes are mutually exclusive and the
    CLI rejects the combination rather than silently picking one.
    """
    if symmetric:
        prompt = SYMMETRIC_TEMPLATE.format(
            article=truncate_article(article),
            strategy_desc=STRATEGY_INSTRUCTIONS[strategy],
            paraphrase=SYMMETRIC_PARAPHRASE_INSTRUCTION,
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

    # Which length bucket each generated row is asked for. In default mode the
    # whole run is one unlabelled bucket, which keeps the loop below identical
    # for both modes.
    #
    # Buckets get DISJOINT slices of the source pool rather than all three
    # lengths of the same article. Three retellings of one story share their
    # names, numbers and phrasing, so putting them in one corpus would seed it
    # with near-duplicates -- and if a train/test split later separated them,
    # the test set would contain paraphrases of training rows. Disjoint thirds
    # cost nothing here (there are far more source articles than we need) and
    # keep the corpus safe to split.
    if args.lengths:
        buckets = list(args.lengths)
        per = args.n // len(buckets)
        plan = []
        for i, b in enumerate(buckets):
            k = per + (args.n - per * len(buckets) if i == len(buckets) - 1 else 0)
            plan += [b] * k
    else:
        plan = [None] * args.n

    # Resume support: don't regenerate what we already have.
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

    # Cursor into the source pool, per bucket. Default mode has one cursor over
    # the whole pool starting at `done`, which is exactly what this loop did
    # before length mode existed. Length mode gives each bucket a contiguous,
    # non-overlapping region so no source article is retold at two lengths.
    if args.lengths:
        region = len(real) // len(buckets)
        cursors = {b: i * region for i, b in enumerate(buckets)}
        limits = {b: (i + 1) * region if i < len(buckets) - 1 else len(real)
                  for i, b in enumerate(buckets)}
        for row in existing:                     # resume where each bucket stopped
            b = row.get("length")
            if b in cursors:
                cursors[b] += 1
    else:
        cursors, limits = {None: done}, {None: len(real)}

    results = list(existing)
    pbar = tqdm(total=args.n, initial=done, desc="Generating")
    # Indexed by how many rows we HAVE, not by how many attempts we've made, so
    # a rejected generation retries that bucket against the next source article
    # instead of costing the corpus a row -- the same "keep going until we have
    # n" behaviour the single-length version had.
    while len(results) < len(plan):
        bucket = plan[len(results)]
        if cursors[bucket] >= limits[bucket]:
            # No source articles left for this bucket. Drop its unfilled slots
            # and carry on with the others rather than spinning forever; the
            # run ends short of --n, which the final count makes visible.
            print(f"\n(source pool exhausted for bucket {bucket!r} -- "
                  f"dropping its remaining slots)")
            plan = plan[:len(results)] + [p for p in plan[len(results):] if p != bucket]
            continue
        article = str(real.iloc[cursors[bucket]]["text"])
        cursors[bucket] += 1
        if len(article.split()) < 30:   # too short to manipulate meaningfully
            continue

        target = LENGTH_SPECS[bucket][0] if bucket else None
        strategy = random.choice(cfg.TRANSFORMATIONS)
        data = None
        for _ in range(args.max_retries):
            data = generate_one(client, article, strategy, bucket,
                                symmetric=args.symmetric)
            if data and quality_ok(article, data["synthetic_article"],
                                   target_words=target):
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
        # Only present in length mode, so the default file keeps its exact
        # existing schema and nothing downstream has to learn a new column.
        if bucket:
            row["length"] = bucket
        if args.symmetric:
            # The altered fact restated in the rewritten article's own wording.
            #
            # WHAT THIS IS NOT: a verbatim pointer into the article. The prompt
            # asks for the sentence word for word, and the model does not comply
            # -- measured on a 30-article pilot, the returned string is an exact
            # substring of the article 10% of the time, 17% after normalising
            # punctuation, and matches some sentence at 80% token overlap only
            # 33% of the time. It writes a plausible restatement instead of
            # copying. Do not use it to locate the edit in the text.
            #
            # WHAT IT IS FOR: a second, differently-worded record of the change,
            # which makes the audit less dependent on lexical luck. Verifying an
            # edit by matching modified_fact against this field succeeds 86.7%
            # of the time against 73.3% matching it against the whole article.
            # That is a real improvement but not a fix -- heavy paraphrase
            # rewords the altered fact itself, so some loss of traceability is
            # the price of removing the edit-distance asymmetry, and the honest
            # figure to quote is the ~99.5% of the lightly-edited corpus falling
            # to roughly 87% here.
            row["modified_fact_as_written"] = data.get("modified_fact_as_written", "")
        results.append(row)
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
    if "length" in df.columns:
        # The point of the run is the length distribution, so report whether it
        # actually landed where it was aimed rather than assuming the LLM obeyed.
        w = df["text"].astype(str).str.split().str.len()
        print("\nLength bucket   n   target   median words")
        for b in df["length"].dropna().unique():
            sel = w[df["length"] == b]
            print(f"  {b:10s} {len(sel):4d}   {LENGTH_SPECS[b][0]:5d}   {int(sel.median()):6d}")


if __name__ == "__main__":
    main()