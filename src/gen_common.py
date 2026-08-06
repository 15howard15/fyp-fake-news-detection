
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
               ratio_range: tuple = (0.4, 2.0)) -> bool:
    """Length-ratio sanity filter: reject if the rewrite is wildly shorter or
    longer than the source, which usually means the model hallucinated,
    refused, or truncated."""
    s_words = len(source.split())
    g_words = len(generated.split())
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
