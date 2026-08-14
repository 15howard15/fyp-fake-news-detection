# Synthetic corpora

The LLM-generated text this project's findings rest on. These files are
committed, unlike everything else under `data/`, because they cannot be
recreated: generation used paid OpenAI calls at `temperature=0.7`, so re-running
the generators produces *different* text and would not reproduce the reported
numbers. `data/raw/` and `data/processed/` are excluded instead — the raw
corpora are public downloads, and the processed splits are rebuilt from raw +
these files by the `build_*_datasets.py` scripts.

All of it was produced with **GPT-4o-mini** (`config.OPENAI_MODEL`).

| File | Rows | What it is | Made by |
|---|---|---|---|
| `synthetic_fake.csv` | 500 | ISOT real articles with a single fact altered — the fake class for the replacement and augmentation recipes | `generate_synthetic_fake.py` |
| `synthetic_fake_mixedlen.csv` | 500 | The same single-fact manipulations written at three very different lengths (~25 / ~100 / ~400 words) over **disjoint** source articles, so the corpus holds no near-duplicate retellings of one story. Backs the length-controlled recipe; `synthetic_fake.csv` is untouched | `generate_synthetic_fake.py --lengths short medium long` |
| `synthetic_fake_liar.csv` | 200 | The same idea sourced from LIAR statements instead of ISOT, for the diverse-sourcing recipe | `generate_synthetic_fake_liar.py` |
| `synthetic_real.csv` | 1000 | ISOT real articles paraphrased with every fact preserved — the "synthetic real" control behind the authorship-shortcut check | `generate_synthetic_real.py` |
| `style_attack.csv` | 200 | Tone-only rewrites of 200 held-out articles: real made sensational, fake made neutral. Labels unchanged | `generate_style_attack.py` |
| `style_attack_originals.csv` | 200 | The untouched versions of those same 200 articles, for the paired before/after comparison | `generate_style_attack.py` |
| `style_attack_reverse.csv` | 200 | The opposite attack — real made neutral, fake made sensational — used to test whether the fix generalizes | `generate_style_attack_reverse.py` |
| `style_attack_reverse_originals.csv` | 200 | Paired originals for the reverse attack | `generate_style_attack_reverse.py` |
| `style_attack_rulebased.csv` | 200 | A non-LLM rule-based tone attack, as a comparison point for the LLM-generated one | `generate_style_attack.py` |
| `counter_style_training.csv` | 200 | Paired tone-shifted twins added to training so tone stops predicting the label — the style-robust fix | `generate_counter_style_training.py` |

## Columns

Most files carry `text`, `label` (0 = real, 1 = fake) and `source`.
Beyond that:

- `source_text` — the original article the row was derived from. **Truncated to
  1,000 characters** in `synthetic_fake.csv`, which is why only some fact edits
  can be verified end to end (see `check_synthetic_quality.py`).
- `modified_fact` — the edit as `original -> altered`, recorded at generation
  time. This is what makes the fact changes auditable after the fact.
- `transformation` — which manipulation was applied (`fact_manipulation`,
  `context_distortion`, `selective_omission`, `tone_adjustment`).
- `length` — `synthetic_fake_mixedlen.csv` only: which length bucket the row was
  generated for (`short` ≈ 25 words, `medium` ≈ 100, `long` ≈ 400). Absent from
  every other file, whose rows follow their source article's length.
- `orig_id` — links an attacked row to its untouched twin in the matching
  `*_originals.csv`.
- `attack_type` — the direction of the tone rewrite.

## Please note

Rows labelled `1` contain **deliberately false statements**, generated for the
purpose of training and stress-testing detection models. They are derived from
ISOT, an existing public fake-news research corpus, and are research material —
not claims about the world, and not for redistribution as news.

Quality was measured rather than assumed; run `python src/check_synthetic_quality.py`
for diversity, fact-change verification and (with `--judge N`) plausibility
ratings.
