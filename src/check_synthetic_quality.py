"""
check_synthetic_quality.py -- evidence that the generated text is actually
usable, not just the right length.

Until now the only gate on generated text was gen_common.quality_ok(), a
length-ratio filter that rejects refusals and truncations. That is enough to
catch a broken API response, but it supports no claim about whether the
synthetic fake news reads plausibly or whether the intended factual change
actually happened. Every result in this project rests on that data being
reasonable, so this script measures it directly.

Three checks, in increasing cost:

1. DIVERSITY (free) -- distinct-n and type-token ratio show whether the
   generator produced 500 varied articles or 500 rewordings of the same few
   templates. Mean pairwise TF-IDF cosine similarity is reported alongside as
   a self-similarity proxy (cheaper than Self-BLEU, same purpose): a corpus
   that is internally near-identical would score high here.

2. FACT-CHANGE VERIFICATION (free) -- generate_synthetic_fake.py records a
   `modified_fact` string of the form "original -> altered" for every article.
   That makes the intended edit checkable after the fact: the original side
   should appear in the source article, the altered side should appear in the
   generated article, and the altered side should NOT already be in the
   source. Anything failing those tests is an article whose "fake" label may
   not correspond to a real factual change.

3. LLM-AS-JUDGE (costs money, opt-in via --judge N) -- asks the model to rate
   a random sample for plausibility as a news article. Off by default because
   it spends OPENAI_API_KEY budget; everything above runs offline.

   IMPORTANT CAVEAT: the judge is cfg.OPENAI_MODEL, the same model that
   generated the text being rated. Models are known to favour their own
   output, so a high score here is weaker evidence than the same score from
   an independent judge or a human sample would be. The real ISOT fake news
   is rated alongside as a reference point, which is what makes the
   comparison interpretable at all -- but the absolute numbers should be
   reported with the shared-model-family caveat attached, not as a neutral
   quality measurement.

Writes results/extra/synthetic_quality.csv.
"""
import argparse
import random
import re
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import config as cfg

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EXTRA_DIR = cfg.RESULTS_DIR / "extra"
EXTRA_DIR.mkdir(parents=True, exist_ok=True)


def distinct_n(texts, n):
    """Fraction of n-grams that are unique. 1.0 = every n-gram appears once
    (maximally varied); values near 0 mean the corpus repeats itself."""
    grams = set()
    total = 0
    for t in texts:
        w = str(t).split()
        for i in range(len(w) - n + 1):
            grams.add(tuple(w[i:i + n]))
            total += 1
    return len(grams) / max(total, 1)


def mean_pairwise_similarity(texts, sample=300, seed=cfg.SEED):
    """Mean off-diagonal TF-IDF cosine similarity over a random sample.
    Self-BLEU proxy -- high values mean the articles resemble each other."""
    rng = random.Random(seed)
    pool = list(texts)
    if len(pool) > sample:
        pool = rng.sample(pool, sample)
    X = TfidfVectorizer(max_features=20_000).fit_transform([str(t) for t in pool])
    S = (X @ X.T).toarray()
    np.fill_diagonal(S, np.nan)
    return float(np.nanmean(S))


def check_diversity(rows):
    print("\n=== 1. DIVERSITY (is the generator repeating itself?) ===")
    print(f"{'file':28s} {'n':>5s} {'distinct-1':>11s} {'distinct-2':>11s} "
          f"{'distinct-3':>11s} {'mean pair sim':>14s}")
    for name in ("synthetic_fake", "synthetic_fake_liar", "synthetic_real"):
        p = cfg.SYNTHETIC_DIR / f"{name}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        texts = df["text"].astype(str).tolist()
        d1, d2, d3 = distinct_n(texts, 1), distinct_n(texts, 2), distinct_n(texts, 3)
        sim = mean_pairwise_similarity(texts)
        print(f"{name:28s} {len(texts):5d} {d1:11.4f} {d2:11.4f} {d3:11.4f} {sim:14.4f}")
        rows.append({"check": "diversity", "file": name, "n": len(texts),
                     "distinct_1": round(d1, 4), "distinct_2": round(d2, 4),
                     "distinct_3": round(d3, 4), "mean_pairwise_sim": round(sim, 4)})

    # Reference point: the same measures on the REAL articles the synthetic
    # ones were derived from. Synthetic text scoring close to real text is the
    # outcome that supports using it as a training substitute.
    p = cfg.PROCESSED_DIR / "isot_fake.csv"
    if p.exists():
        real = pd.read_csv(p)["text"].astype(str).head(500).tolist()
        d1, d2, d3 = distinct_n(real, 1), distinct_n(real, 2), distinct_n(real, 3)
        sim = mean_pairwise_similarity(real)
        print(f"{'[reference] real ISOT fake':28s} {len(real):5d} {d1:11.4f} "
              f"{d2:11.4f} {d3:11.4f} {sim:14.4f}")
        rows.append({"check": "diversity", "file": "reference_real_isot_fake",
                     "n": len(real), "distinct_1": round(d1, 4),
                     "distinct_2": round(d2, 4), "distinct_3": round(d3, 4),
                     "mean_pairwise_sim": round(sim, 4)})


def _norm(s):
    """Loose match -- lowercase, collapse whitespace, drop punctuation that the
    model commonly re-styles (quotes, commas) so a genuine match isn't missed
    on formatting alone."""
    s = str(s).lower()
    s = re.sub(r"[‘’“”\"',]", "", s)
    return re.sub(r"\s+", " ", s).strip()


_STOP = set("a an the of in on at to for and or but is are was were be been "
            "that this it its as by with from has have had will would".split())


def _content_words(s):
    return {w for w in re.findall(r"[a-z0-9]+", _norm(s)) if w not in _STOP}


def _fuzzy_in(needle, haystack, threshold=0.6):
    """Does `needle` appear in `haystack` in substance, if not verbatim?

    Exact substring matching badly understates the ISOT-sourced set: the model
    reports the original fact in its own words rather than quoting the article
    verbatim, so a real, correctly-recorded edit still fails a strict test.
    This asks the weaker question actually of interest -- are most of the
    content words of the fact present -- and is reported ALONGSIDE the strict
    figure rather than replacing it, since it is the looser evidence of the
    two."""
    nw = _content_words(needle)
    if not nw:
        return False
    return len(nw & _content_words(haystack)) / len(nw) >= threshold


def check_fact_changes(rows):
    print("\n=== 2. FACT-CHANGE VERIFICATION (did the intended edit happen?) ===")
    for name in ("synthetic_fake", "synthetic_fake_liar"):
        p = cfg.SYNTHETIC_DIR / f"{name}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "modified_fact" not in df.columns or "source_text" not in df.columns:
            print(f"  {name}: no modified_fact/source_text columns, skipping")
            continue

        parsed = 0
        strict = {"orig": 0, "new": 0, "novel": 0, "all": 0}
        fuzzy = {"orig": 0, "new": 0, "novel": 0, "all": 0}
        for _, r in df.iterrows():
            mf = str(r.get("modified_fact", ""))
            if "->" not in mf:
                continue
            parsed += 1
            old, new = [x.strip() for x in mf.split("->", 1)]
            src_raw, gen_raw = r["source_text"], r["text"]
            src, gen = _norm(src_raw), _norm(gen_raw)

            s_o, s_n, s_v = _norm(old) in src, _norm(new) in gen, _norm(new) not in src
            strict["orig"] += s_o; strict["new"] += s_n; strict["novel"] += s_v
            strict["all"] += (s_o and s_n and s_v)

            f_o = _fuzzy_in(old, src_raw)
            f_n = _fuzzy_in(new, gen_raw)
            f_v = not _fuzzy_in(new, src_raw)
            fuzzy["orig"] += f_o; fuzzy["new"] += f_n; fuzzy["novel"] += f_v
            fuzzy["all"] += (f_o and f_n and f_v)

        n = max(parsed, 1)
        # Each criterion needs the matching standard that actually fits it,
        # not one standard applied uniformly:
        #   - "original traceable" and "altered present" SHOULD use the loose
        #     test, because the model records the fact in its own words rather
        #     than quoting the article, so a verbatim test understates them.
        #   - "altered is genuinely new" MUST use the strict test. Only one
        #     fact changes per article, so the altered version shares nearly
        #     all its content words with the original ("$31 billion over the
        #     next decade" vs "$50 billion over the next decade"); a loose test
        #     would call that "already present in the source" and wrongly count
        #     a real edit as a non-edit.
        # The combined figure therefore uses loose/loose/strict, which is why
        # it is higher than a naive intersection of all three loose or all
        # three strict counts.
        combined = 0
        for _, r in df.iterrows():
            mf = str(r.get("modified_fact", ""))
            if "->" not in mf:
                continue
            old, new = [x.strip() for x in mf.split("->", 1)]
            if (_fuzzy_in(old, r["source_text"]) and _fuzzy_in(new, r["text"])
                    and _norm(new) not in _norm(r["source_text"])):
                combined += 1

        print(f"  {name} ({parsed} of {len(df)} rows had a parseable 'old -> new'):")
        print(f"    {'':44s} {'verbatim':>10s} {'substance':>10s}")
        for key, label in (("orig", "original fact traceable to source article"),
                           ("new", "altered fact present in generated article"),
                           ("novel", "altered fact is genuinely new vs source")):
            print(f"    {label:44s} {100*strict[key]/n:9.1f}% {100*fuzzy[key]/n:9.1f}%")
        print(f"    {'EDIT VERIFIED (loose/loose/strict, see code)':44s} "
              f"{'':10s} {100*combined/n:9.1f}%")

        # That headline figure can be dragged down by an artefact of the audit
        # trail rather than by bad generation. HISTORICALLY, the generators
        # stored only truncate_article(article, 1000) in `source_text` -- a
        # 1,000-character preview -- while ISOT articles typically run
        # 1,500-2,500 characters. When the altered fact came from later in the
        # article, the evidence needed to verify it was never saved, so a real
        # edit registered as "unverifiable".
        #
        # The generators now store up to cfg.FULL_SOURCE_CAP (10,000) chars,
        # which is effectively the whole article, so data generated AFTER that
        # change has no truncation artefact and this split simply reports ~100%
        # complete. The split is kept because older corpora (and the committed
        # results) still carry the 1,000-char previews; it separates "the edit
        # failed" from "the edit cannot be checked", which are very different
        # claims. A source_text counts as "truncated" only if its length sits
        # exactly AT a known cap boundary (historic 1000, or current
        # cfg.FULL_SOURCE_CAP). We test against those known caps rather than the
        # data's own max -- otherwise, in a fully complete corpus, the single
        # longest genuine article would be mislabelled as "truncated" and
        # everything shorter as "complete", inventing a split that isn't real.
        # New-cap data therefore has zero truncated rows and the split below
        # simply doesn't run, which is correct.
        src_len = df["source_text"].astype(str).str.len()
        caps = {1000, getattr(cfg, "FULL_SOURCE_CAP", 10_000)}
        complete = ~src_len.isin(caps)
        if complete.any() and (~complete).any():
            tr = pd.Series(
                [_fuzzy_in(str(m).split("->")[0], s)
                 for m, s in zip(df["modified_fact"], df["source_text"])],
                index=df.index)
            c_pct = 100 * tr[complete].mean()
            t_pct = 100 * tr[~complete].mean()
            print(f"    -> where the FULL source was saved (n={int(complete.sum())}): "
                  f"{c_pct:.1f}% traceable")
            print(f"    -> where it was truncated        (n={int((~complete).sum())}): "
                  f"{t_pct:.1f}% traceable (evidence not saved, not necessarily a failed edit)")
            rows.append({"check": "fact_change_truncation_split", "file": name,
                         "n": int(complete.sum()),
                         "traceable_full_source_pct": round(c_pct, 1),
                         "traceable_truncated_source_pct": round(t_pct, 1)})
        rows.append({"check": "fact_change", "file": name, "n": parsed,
                     "edit_verified_pct": round(100*combined/n, 1),
                     "orig_in_source_pct": round(100*strict["orig"]/n, 1),
                     "new_in_generated_pct": round(100*strict["new"]/n, 1),
                     "new_not_in_source_pct": round(100*strict["novel"]/n, 1),
                     "fully_verified_pct": round(100*strict["all"]/n, 1),
                     "orig_in_source_fuzzy_pct": round(100*fuzzy["orig"]/n, 1),
                     "new_in_generated_fuzzy_pct": round(100*fuzzy["new"]/n, 1),
                     "new_not_in_source_fuzzy_pct": round(100*fuzzy["novel"]/n, 1),
                     "fully_verified_fuzzy_pct": round(100*fuzzy["all"]/n, 1)})


JUDGE_SYSTEM = (
    "You rate news article excerpts for surface plausibility only. You are NOT "
    "judging whether the claims are true. Reply with JSON: "
    '{"plausible": 1-5, "reason": "<10 words"} where 5 means it reads exactly '
    "like a professionally written news article and 1 means it is obviously "
    "machine-generated, incoherent, or not a news article at all."
)


def check_llm_judge(rows, n_samples):
    print(f"\n=== 3. LLM-AS-JUDGE (plausibility, {n_samples} samples -- SPENDS API BUDGET) ===")
    import json
    from openai import OpenAI
    from gen_common import truncate_article

    client = OpenAI()
    rng = random.Random(cfg.SEED)

    for name in ("synthetic_fake", "isot_fake_reference"):
        if name == "synthetic_fake":
            p = cfg.SYNTHETIC_DIR / "synthetic_fake.csv"
        else:
            p = cfg.PROCESSED_DIR / "isot_fake.csv"
        if not p.exists():
            continue
        texts = pd.read_csv(p)["text"].astype(str).tolist()
        sample = rng.sample(texts, min(n_samples, len(texts)))

        scores = []
        for t in sample:
            try:
                resp = client.chat.completions.create(
                    model=cfg.OPENAI_MODEL,
                    messages=[{"role": "system", "content": JUDGE_SYSTEM},
                              {"role": "user", "content": truncate_article(t, 3000)}],
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                scores.append(int(json.loads(resp.choices[0].message.content)["plausible"]))
            except Exception as e:
                print(f"    (skipped one: {e})")
        if scores:
            arr = np.array(scores)
            print(f"  {name:22s} n={len(arr):3d} mean={arr.mean():.2f} "
                  f"median={np.median(arr):.1f} pct_4_or_5={100*(arr>=4).mean():.1f}%")
            rows.append({"check": "llm_judge", "file": name, "n": len(arr),
                         "mean_plausibility": round(float(arr.mean()), 2),
                         "pct_4_or_5": round(float(100*(arr >= 4).mean()), 1)})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge", type=int, default=0, metavar="N",
                    help="also run the LLM plausibility judge on N random samples "
                         "per corpus (default 0 = off; this SPENDS OpenAI budget)")
    args = ap.parse_args()

    rows = []
    check_diversity(rows)
    check_fact_changes(rows)
    if args.judge:
        check_llm_judge(rows, args.judge)
    else:
        print("\n=== 3. LLM-AS-JUDGE: skipped (pass --judge 50 to run; costs API budget) ===")

    out = EXTRA_DIR / "synthetic_quality.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()