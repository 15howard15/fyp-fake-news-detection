# Code walkthrough — every file, and where to find things fast

Companion to `PRESENTATION_SCRIPT.md`. That one covers the *results*; this one
covers the *system*.

**Built for live questioning.** Every entry gives the file, the line number, and
the exact command. If your supervisor says "show me where you do X", use the
jump table below — don't scroll hunting for it.

---

## ⚡ Jump table — "show me where you…"

| They ask | File · line | Open with |
|---|---|---|
| …clean the text | `preprocessing.py` **L39** `clean_text` | `code -g src/preprocessing.py:39` |
| …compute the metrics | `metrics.py` **L17** `compute_metrics` | `code -g src/metrics.py:17` |
| …train LR and SVM | `train.py` **L72** `train_lr_svm` | `code -g src/train.py:72` |
| …train the CNN | `train.py` **L201** `train_cnn` | `code -g src/train.py:201` |
| …train BERT | `train.py` **L269** `train_bert` | `code -g src/train.py:269` |
| …set the random seed | `repro.py` **L8** `set_determinism` | `code -g src/repro.py:8` |
| …split train from test | `build_test_sets.py` **L12** `main` | `code -g src/build_test_sets.py:12` |
| …build the training recipes | `build_core_datasets.py` **L12** `main` | `code -g src/build_core_datasets.py:12` |
| …call the OpenAI API | `gen_common.py` **L38** `call_llm` | `code -g src/gen_common.py:38` |
| …filter bad generations | `gen_common.py` **L25** `quality_ok` | `code -g src/gen_common.py:25` |
| …check for data leakage | `evaluate.py` **L661** `cmd_leakage` | `code -g src/evaluate.py:661` |
| …find the misclassified articles | `evaluate.py` **L480** `cmd_hard_examples` | `code -g src/evaluate.py:480` |
| …test on WELFake | `evaluate.py` **L380** `cmd_cross_target` | `code -g src/evaluate.py:380` |
| …run the seed experiment | `run_multiseed_robustness.py` **L60** `run_cnn`, **L104** `run_bert` | `code -g src/run_multiseed_robustness.py:60` |
| …evaluate the style attack | `eval_style_robustness.py` **L185** `main` | `code -g src/eval_style_robustness.py:185` |
| …verify the synthetic data quality | `check_synthetic_quality.py` **L153** `check_fact_changes` | `code -g src/check_synthetic_quality.py:153` |
| …set a hyperparameter | `config.py` (whole file, 110 lines) | `code -g src/config.py:1` |
| …build the report | `build_report.py` **L364** `collect`, **L388** `main` | `code -g src/build_report.py:364` |

*(In VS Code, `code -g file:line` jumps straight to that line. Or **Ctrl+P**,
type the filename, then **Ctrl+G** and the number.)*

> **Before presenting, run `python src/build_code_map.py`.** It re-reads `src/`
> and corrects every line number in this table. They shift whenever a file
> changes — a formatter stripping comments once moved every function in
> `train.py` by about 26 lines — and a wrong number during a live question is
> worse than no table.

---

## The one-paragraph opener

> "`src/` has 32 Python files. Five are shared libraries that never run on their
> own; the other twenty-seven are pipeline steps you run in order. Every filename
> starts with its job — `load_`, `generate_`, `build_`, `train`, `run_`, `eval` —
> so the structure is visible in a plain file listing without needing a diagram."

---

## The shape of it

Data flows one direction. Each stage reads only what the previous stage wrote:

```
raw datasets                      (downloaded manually)
   ↓  load_data.py
data/processed/*.csv              normalised, one format
   ↓  generate_*.py               ← the only steps that cost money
data/synthetic/*.csv              LLM-written text
   ↓  build_*_datasets.py
data/processed/train_*.csv        the training recipes
   ↓  train.py
models/                           trained checkpoints
   ↓  evaluate.py / run_*.py
results/*.json, results/extra/    every number
   ↓  build_report.py
results_report.html               the report
```

**If asked why so many files:** each does exactly one thing — one dataset
variant, one generation strategy, one experiment. Any stage can be re-run
without touching the others, and a failure is localised. The alternative is
three huge scripts with flags controlling which half runs.

---

## The five shared libraries (never run directly)

This group is your answer to *"how do you know the models are compared fairly?"*

### `config.py` — 110 lines, no functions

Every path, the seed, every hyperparameter. Nothing else hard-codes a path or a
learning rate.

**Worth knowing by heart:**

| Setting | Value |
|---|---|
| `SEED` | 42 |
| `TEST_SIZE` | 0.20 |
| `TFIDF_MAX_FEATURES` | 50,000 |
| `TFIDF_NGRAM_RANGE` | (1, 2) — single words and word pairs |
| `CNN_MAX_LEN` | 300 tokens |
| `BERT_MAX_LEN` | 512 tokens |
| `BERT_LR` | 2e-5 |
| `BERT_EPOCHS` | 3 |
| `OPENAI_MODEL` | gpt-4o-mini |

### `preprocessing.py` — L39 `clean_text`, L67 `clean_series`

> "One cleaning pipeline used by all four models. If each model cleaned text its
> own way, a difference in results could be preprocessing rather than the model —
> that would invalidate the whole comparison."

`aggressive=True` (TF-IDF, CNN): lowercase, strip URLs/HTML/punctuation, drop
stopwords. `aggressive=False` (BERT): minimal — its own tokeniser handles the rest.

### `metrics.py` — L17 `compute_metrics`, L42 `save_metrics`

All five metrics computed identically everywhere, written to
`results/metrics_<MODEL>_<recipe>.json` in one consistent format.

### `gen_common.py` — L20 `truncate_article`, L25 `quality_ok`, L38 `call_llm`

Shared OpenAI plumbing and prompts. **`quality_ok` (L25) is worth flagging
honestly**: it is only a length-ratio filter — it catches refusals and
truncations, nothing about whether the text is good. That is why
`check_synthetic_quality.py` exists.

### `repro.py` — L8 `set_determinism`

Seeds Python, NumPy and Torch, sets `CUBLAS_WORKSPACE_CONFIG`, pins
`cudnn.deterministic = True` and `benchmark = False`. Called at the top of every
neural training function.

> "Without the cuDNN flags, the same seed can still give different results on a
> GPU because it picks a different algorithm each run. I added this after
> discovering that."

---

## Stage 1 — Load

```bash
python src/load_data.py     # L6 load_isot · L36 load_liar · L72 load_welfake
python src/eda.py
```

*Writes:* `isot_real`, `isot_fake`, `liar_real`, `liar_fake`, `welfake_real`,
`welfake_fake` → `data/processed/`, and `results/eda/`

The EDA output is where "LIAR ≈ 17 words, ISOT ≈ 380 words" comes from.

---

## Stage 2 — Generate *(the only steps that spend money)*

| Command | Produces |
|---|---|
| `python src/generate_synthetic_fake.py --n 500` | Real article, one fact altered |
| `python src/generate_synthetic_real.py --n 1000` | Paraphrased real articles — **the control** |
| `python src/generate_synthetic_fake_liar.py --n 200` | Same idea, LIAR-sourced |
| `python src/generate_style_attack.py --n_per_class 100` | Tone-rewritten test set (RQ4 attack) |
| `python src/generate_style_attack_reverse.py --n_per_class 100` | The opposite tone direction |
| `python src/generate_counter_style_training.py --n_per_class 100` | Tone twins **for training** — the RQ4 fix |

> "Generation runs at temperature 0.7, so it's non-deterministic — re-running
> produces different text and wouldn't reproduce my numbers. That's exactly why
> `data/synthetic/` is version-controlled while the rest of `data/` isn't."

---

## Stage 3 — Assemble the recipes

No machine learning here — just combining rows, holding size constant.

| Command | Builds |
|---|---|
| `python src/build_core_datasets.py` | `real_real`, `mixed`, `real_syn` |
| `python src/build_augmented_datasets.py` | augmentation variants |
| `python src/build_synthetic_real_datasets.py` | `c2_synreal_realfake`, `c3_synreal_synfake` |
| `python src/build_swap_sweep_datasets.py` | the 0→100% synthetic sweep |
| `python src/build_multisource_dataset.py` | `real_syn_multisource` |
| `python src/build_style_robust_dataset.py` | `style_robust` |
| `python src/build_test_sets.py` | `test_indomain`, `test_crossdomain`, `test_crossdomain2` |

**`build_test_sets.py` (L12) is the one to explain properly:**

> "This is where the evaluation design lives. The real-news side is held-out ISOT
> in *every* test set — only the fake side changes. So when I compare LIAR against
> WELFake I'm changing one variable, not two."

---

## Stage 4 — Train

**`train.py`** — one entry point, 408 lines.

```bash
python src/train.py --model all
python src/train.py --model bert --dataset style_robust
python src/train.py --model all --dataset real_real --max-words 300
```

| Line | Function | What it does |
|---|---|---|
| L72 | `train_lr_svm` | TF-IDF, then LR and SVM from the same matrix |
| L112 | `Vocab` | CNN vocabulary builder |
| L156 | `TextCNN` | The CNN architecture |
| L174 | `get_cnn_vocab_and_embed` | GloVe loading + the `.clone()` fix |
| L201 | `train_cnn` | |
| L269 | `train_bert` | Includes warmup, decay, gradient clipping |
| L344 | `main` | Argument parsing and dispatch |

| Flag | Does |
|---|---|
| `--model` | `lr_svm`, `cnn`, `bert`, `all` |
| `--dataset` | Any recipe with a `train_<name>.csv` |
| `--seed` | Repeat run, saved under a separate label |
| `--max-words N` | Cap every document for **all** models |
| `--grad_accum` | If BERT runs out of VRAM |

**Three things worth pointing at if they open this file:**

**L72 — why LR and SVM share a function:** they use the same TF-IDF matrix, so
fitting it once guarantees they see identical features.

**L174 — the `.clone()` comment:** `nn.Embedding.from_pretrained(..., freeze=False)`
shares memory with the tensor you hand it. Without cloning, the first
composition's fine-tuning would leak into every later one's "fresh" start.

> "That's a bug I found and fixed. It would have silently corrupted every CNN
> result after the first."

**L269 — the BERT stability recipe:** warmup over 10% of steps, linear decay,
gradient clipping at 1.0. Fine-tuning a pre-trained transformer at a flat
learning rate from step one is a known source of early instability.

---

## Stage 5 — Evaluate

**`evaluate.py`** — 1,035 lines, eight subcommands.

| Line | Subcommand | Answers |
|---|---|---|
| L66 | `master` | Collect all metrics into one table |
| L351 | `error-analysis` | Confusion matrices, vocabulary overlap |
| L380 | `cross-target` | Re-score existing checkpoints on WELFake |
| L480 | `hard-examples` | **The articles a model got most confidently wrong** |
| L580 | `length-sweep` | Truncate the test set — separate length from domain |
| L661 | `leakage` | Train/test overlap + corpus independence |
| L790 | `seed-summary` | Mean ± SD across seeds |
| L827 | `case-studies` | Concrete style-attack flips with article text |

**The three to name in a review:**

**L661 `cmd_leakage`** — found that ISOT contains 23.7% duplicate fake articles,
and that 63.8% of WELFake also exists in ISOT. Fails above a 2% threshold.

**L480 `cmd_hard_examples`** — prints the most confidently-wrong predictions.
This produced the two misclassified Reuters articles behind the authorship
finding.

**L580 `cmd_length_sweep`** — shortens test articles to separate a length effect
from a domain effect.

**The dedicated runners:**

| Command | Answers |
|---|---|
| `python src/run_swap_sweep_experiment.py` | RQ2 — the 0→100% curve |
| `python src/run_multiseed_robustness.py --comps <names>` | RQ3 — 3 seeds per condition |
| `python src/eval_style_robustness.py` | RQ4 — flip rates, **no retraining** |
| `python src/check_synthetic_quality.py [--judge 50]` | Data quality |
| `python src/run_train_extra_experiments.py` | LR/SVM on augmentation variants |
| `python src/run_deep_extra_experiments.py --models cnn bert` | Same for deep models |

> "`eval_style_robustness.py` doesn't retrain anything — it loads existing
> checkpoints and scores them on the attacked articles. Same model, two versions
> of the same text. That's what makes the RQ4 comparison fair."

---

## Stage 6 — Report

```bash
python src/export_detector_model.py    # LR weights → detector_model.js
python src/build_report.py             # → results_report.html
python src/build_walkthrough.py        # → pipeline_walkthrough.ipynb
python src/verify_detector.py          # browser scorer vs sklearn (needs Node)
```

**`build_report.py` (423 lines)** — one function per report section:

| Line | Function | Feeds |
|---|---|---|
| L56 | `liar_block` | RQ1 main chart |
| L70 | `welfake_block` | Framework tab |
| L87 | `sweep_block` | RQ2 |
| L106 | `style_block` | RQ4 |
| L171 | `families_block` | RQ3 cards + Evidence 1 |
| L248 | `seed_runs_block` | RQ3 seed table |
| L287 | `quality_block` | Data quality |
| L334 | `demo_examples` | Try-it examples |
| L364 | `collect` | **Assembles everything** |

**The point to make here:**

> "The report is *generated*, not written. Every number is read from `results/`
> when the page is built, so the page can't disagree with the data. It used to be
> hand-written HTML with figures typed in, and they drifted out of sync more than
> once — so I removed the possibility rather than trying to be careful."

---

## If they ask you to run something live

**Safe and fast (seconds, read-only):**

```bash
python src/build_report.py            # ~2s, rebuilds the whole report
python src/check_synthetic_quality.py # ~20s, diversity + fact verification
python src/evaluate.py leakage        # ~30s, the overlap findings
python src/evaluate.py seed-summary   # instant, reads the saved CSV
```

**Do NOT run live** — these take minutes to hours:

```bash
python src/train.py --model all       # hours on a GPU
python src/generate_*.py              # spends real API budget
python src/run_multiseed_robustness.py # ~20 minutes
```

If asked to demonstrate training, say:

> "That's a few hours on a GPU, so I've got the saved checkpoints instead — but
> I can show you the training code and the metrics it produced."

---

## Code-review questions

**"Why 32 files instead of a few?"**
> "Each does one thing, so any stage re-runs without touching the others and a
> failure is localised. The prefixes make the grouping visible in a file listing."

**"How do you know the models are compared fairly?"**
> "Three shared modules. `preprocessing.py` means all four see identically cleaned
> text, `metrics.py` means all four are scored identically, `config.py` means the
> settings live in one place."

**"Can I reproduce this?"**
> "Code and synthetic data, yes — `requirements.txt` is pinned exactly and
> `requirements-lock.txt` has all 104 packages. Training is a few hours on a GPU.
> What you can't reproduce exactly is generation, because it's non-deterministic —
> which is why that data is committed rather than regenerated."

**"What if I run the scripts out of order?"**
> "They fail with a message naming the script you need first. Each checks its
> inputs exist before doing any work."

**"Is any of it tested?"**
> "Not unit tests — that's a fair criticism. What it has instead are
> pipeline-level checks: `evaluate.py leakage` fails if train and test overlap,
> `verify_detector.py` checks the browser model against sklearn, and
> `export_detector_model.py` asserts the vectoriser settings it depends on. Those
> catch the failures that would actually invalidate results."

**"Show me a bug you found."**
> Open `train.py` **L174**. The embedding-sharing bug — without the `.clone()`,
> every CNN result after the first would have been silently corrupted by the
> previous composition's fine-tuning.

---

## Three sentences to remember

1. **32 files, six stages, one direction** — each stage reads only what the
   previous one wrote.
2. **Five shared modules mean no experiment differs in the plumbing** — cleaning,
   metrics and seeding are shared code, not repeated code.
3. **The report and the notebook are generated from the results files**, so they
   cannot disagree with the data.
