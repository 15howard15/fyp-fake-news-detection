
"""Shared helpers and prompts for the OpenAI-based generation scripts."""
import json
import time

import config as cfg


def truncate_article(text: str, max_chars: int = 4000) -> str:
    """Keep prompt size (and cost) bounded."""
    return text[:max_chars]


def quality_ok(source: str, generated: str, min_words: int = 20,
               ratio_range: tuple = (0.4, 2.0),
               target_words: int = None, target_tol: float = 0.6) -> bool:
    s_words = len(source.split())
    g_words = len(generated.split())

    if target_words is not None:
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
    except Exception as e:
        print(f"  API error: {e}")
        time.sleep(5)
        return None
    return None


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
