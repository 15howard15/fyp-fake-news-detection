
"""Shared helpers and prompts for the OpenAI-based data-generation scripts
(generate_synthetic_fake.py, generate_synthetic_real.py,
generate_synthetic_fake_liar.py, generate_style_attack.py,
generate_counter_style_training.py, generate_style_attack_reverse.py).
truncate_article/quality_ok/the generate_one API-call-and-retry pattern were
copy-pasted near-identically into all six scripts, and the three style-attack
scripts additionally shared a verbatim system prompt (two of them also shared
both tone-transfer templates verbatim) -- same class of duplication that
justified train.py/evaluate.py, extracted here instead. Each script keeps its
own remaining prompts/templates and CLI entry point; only genuinely shared
content lives here.
"""
import json
import time

import config as cfg


def truncate_article(text: str, max_chars: int = 4000) -> str:
    """Keep prompt size (and cost) bounded. 4000 chars ~ 1000 tokens."""
    return text[:max_chars]


def quality_ok(source: str, generated: str, min_words: int = 20,
               ratio_range: tuple = (0.4, 2.0),
               target_words: int = None, target_tol: float = 0.6) -> bool:
    """Length sanity filter: reject rewrites whose length says the model
    hallucinated, refused, or truncated.

    Two modes, because the two generation designs mean different things by
    "wrong length":

    - Default (target_words=None): judge the rewrite RELATIVE TO ITS SOURCE.
      The single-edit design keeps the rest of the wording near-identical, so a
      rewrite far from the source length is a failed generation.

    - Length-controlled (target_words set): judge against the length we ASKED
      for. Condensing a 400-word article into a 20-word snippet is a ratio of
      0.05 and the default band would reject every one of them -- the filter
      would silently throw away exactly the short data the experiment exists to
      produce. Here the source ratio carries no information, so it is not used.
      target_tol is the accepted band around the target. A float means a
      symmetric fraction either side (0.6 = accept 40%-160% of the requested
      length), wide enough that the LLM's habitual imprecision about word counts
      doesn't cost most of the batch. A (lo, hi) pair gives an ASYMMETRIC band as
      fractions OF the target (0.85, 1.6 = accept 85%-160%).

      The asymmetric form exists because the error is not symmetric in practice:
      asked to match a source's length, the model reliably undershoots and
      almost never overshoots. A symmetric +/-35% band around a 369-word target
      accepts 240 words, and the run duly came back at a median of 275 against
      the real class's 367 -- close enough to pass the filter, far enough to let
      a word-counter separate the classes at AUC 0.42. Rejecting compression
      harder than expansion is what actually holds the two classes level.
    """
    s_words = len(source.split())
    g_words = len(generated.split())

    if target_words is not None:
        # min_words would reject every valid ~20-word snippet, so the target
        # band replaces it rather than being applied on top.
        if isinstance(target_tol, (tuple, list)):
            lo_f, hi_f = target_tol
        else:
            lo_f, hi_f = 1 - target_tol, 1 + target_tol
        lo = max(1, int(target_words * lo_f))
        hi = int(target_words * hi_f)
        return lo <= g_words <= hi

    if g_words < min_words:
        return False
    ratio = g_words / max(s_words, 1)
    return ratio_range[0] <= ratio <= ratio_range[1]


def call_llm(client, system_prompt: str, prompt: str, response_key: str):
    """Call the chat completion API once, expecting a JSON object containing
    `response_key`. Returns the full parsed dict (callers that need extra
    keys, e.g. 03's fact_table/modified_fact, can still read them) or None on
    failure/refusal/rate-limit."""
    try:
        resp = client.chat.completions.create(
            model=cfg.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        if response_key in data and data[response_key].strip():
            return data
    except json.JSONDecodeError:
        return None
    except Exception as e:  # network / rate limit
        print(f"  API error: {e}")
        time.sleep(5)
        return None
    return None


# ----------------------------------------------------------------------
# Symmetric-edit generation (generate_synthetic_fake.py --symmetric and
# generate_synthetic_real.py --symmetric)
# ----------------------------------------------------------------------
# The default generators edit the two classes by very different amounts. Measured
# with difflib over 120 articles against the 4,000-character window the prompt
# actually sees: synthetic FAKE retains 65.9% of the source verbatim (it alters
# one fact and is told to leave the rest alone), synthetic REAL retains 44.0%
# (it is told to rewrite throughout). A 21.8-point gap in how heavily the two
# classes were rewritten is a feature the model can read INSTEAD of the facts --
# "closer to newswire wording" becomes a proxy for "fake". That is the C3
# authorship shortcut.
#
# The fix is to give both classes the SAME paraphrase instruction and let them
# differ in exactly one thing: whether a fact was altered. Writing the shared
# half once, here, is what makes that claim checkable -- two separately-worded
# prompts that merely sound similar would leave the symmetry asserted rather
# than enforced.
# Calibrated against a 30-article pilot per class rather than guessed. The first
# wording included "keep roughly the same length and the same order of events",
# which held the rewrite back to 0.65 similarity -- LESS rewriting than the
# original synthetic-real prompt achieved (0.44). Dropping that clause and
# adding the explicit four-word reuse limit is what brings both classes down to
# the 0.44 target. The clause about presentation order is deliberate: reordering
# is most of what separates a genuine rewrite from a synonym swap.
SYMMETRIC_PARAPHRASE_INSTRUCTION = (
    "Write the story again from scratch, as a different journalist working from "
    "the same notes would. Do not reuse any phrase of more than three "
    "consecutive words from the source. Open on a different sentence from the "
    "source's opening and present the information in a different order. Rebuild "
    "every sentence: switch between active and passive voice, reorder the "
    "clauses, split long sentences and merge short ones, and replace the nouns, "
    "verbs and connectives with alternatives wherever the meaning survives. Use "
    "different reporting verbs throughout. At the sentence level nothing should "
    "be recognisable as copied; at the fact level it must be the same story."
)

SYMMETRIC_SYSTEM_BASE = (
    "You are a data-generation tool for academic fake-news-detection research. "
    "You rewrite news articles in your own words. Rewriting depth must be the "
    "same on every article you are given, regardless of what else you are asked "
    "to do, because the rewrite depth is a controlled variable in this "
    "experiment. Respond ONLY with valid JSON, no markdown, no commentary."
)


# Shared by generate_style_attack.py, generate_counter_style_training.py, and
# generate_style_attack_reverse.py -- all three were found to have this exact
# system prompt copy-pasted verbatim; the two forward-direction scripts also
# shared both tone-transfer templates verbatim. generate_style_attack_reverse.py
# keeps its own two templates (genuinely different wording -- opposite tone
# direction) but reuses this system prompt.
STYLE_TRANSFER_SYSTEM_PROMPT = (
    "You are a controlled text-style-transfer tool for academic robustness "
    "testing of fake-news detectors. You rewrite an article's TONE ONLY. "
    "You must not add, remove, or change any fact, name, date, number, or "
    "claim -- the article must remain exactly as true or false as the "
    "original. Respond ONLY with valid JSON, no markdown, no commentary."
)

SENSATIONALIZE_REAL_TEMPLATE = """Source article (this is REAL, factual news):
\"\"\"{article}\"\"\"

Rewrite this article using sensational, alarmist, emotionally charged
language -- as if a tabloid or clickbait outlet were reporting the SAME
true story. Exaggerate tone and framing only. Do NOT change, add, or
remove any fact, name, date, or number. The rewritten article must still
be 100% factually accurate.

Return JSON with exactly these keys:
{{
  "styled_article": "the full rewritten article text"
}}"""

NEUTRALIZE_FAKE_TEMPLATE = """Source article (this is FAKE / false news):
\"\"\"{article}\"\"\"

Rewrite this article using calm, neutral, measured, factual-sounding
journalistic tone -- as if a sober, credible newsroom were reporting it.
Remove any sensational or emotional language. Do NOT change, add, or
remove any fact, name, date, or number, and do NOT correct or fix the
false claim -- the rewritten article must remain exactly as FALSE as the
original, just calmer in tone.

Return JSON with exactly these keys:
{{
  "styled_article": "the full rewritten article text"
}}"""
