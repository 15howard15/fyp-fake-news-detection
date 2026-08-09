# Code walkthrough — every file, what it does, how it runs

A companion to `PRESENTATION_SCRIPT.md`. That one covers the *results*; this one
covers the *system*, which is what a code review actually asks about.

**The one-sentence version to open with:**

> "`src/` has 32 Python files. Five are shared libraries that never run on
> their own, and the other twenty-seven are pipeline steps you run in order. Every filename
> starts with its job — `load_`, `generate_`, `build_`, `train`, `run_`, `eval`
> — so the structure is visible in a plain file listing without needing a
> diagram."

---

## The shape of it

Data flows in one direction. Each stage only reads what the previous stage
wrote, which is why the run order in the README matters:

```
raw datasets                      (you download these)
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

**Say this if asked why so many files:** each one does exactly one thing — one
dataset variant, one generation strategy, one experiment. That is what makes
them individually testable and lets a single stage be re-run without touching
the others. The alternative is three enormous scripts with flags controlling
which half runs.

---

## The five shared libraries (never run directly)

These are imported by everything else. If a reviewer asks "how do you know the
models are compared fairly?", the answer is in this group.

| File | What it holds | Why it exists |
|---|---|---|
| `config.py` | All paths, the seed, every hyperparameter | One place to change a setting. Nothing else hard-codes a path or a learning rate. |
| `preprocessing.py` | `clean_text()` / `clean_series()` | **One cleaning pipeline used by all four models.** If each model cleaned text its own way, a difference in results could be preprocessing rather than the model — that would invalidate the whole comparison. |
| `metrics.py` | `compute_metrics()`, `save_metrics()` | Accuracy, precision, recall, F1, AUC-ROC computed identically everywhere, and written to `results/metrics_<MODEL>_<recipe>.json` in one consistent format. |
| `gen_common.py` | Shared OpenAI plumbing and prompts | The six generation scripts had this copy-pasted. Now a fix to the retry logic or a prompt happens once. |
| `repro.py` | `set_determinism(seed)` | Seeds Python, NumPy and Torch, and pins cuDNN to deterministic algorithms. Called at the top of every neural training function. |

> "The point of this group is that no experiment can accidentally differ from
> another in the plumbing. Cleaning, metrics and seeding are shared code, not
> repeated code."

---

## Stage 1 — Load

**`load_data.py`** — reads the raw downloads, normalises the labels
(0 = real, 1 = fake), and writes six clean files.

```bash
python src/load_data.py
```
*Writes:* `isot_real`, `isot_fake`, `liar_real`, `liar_fake`, `welfake_real`, `welfake_fake` (all in `data/processed/`)

**`eda.py`** — exploratory analysis: class balance, article lengths, top words.

```bash
python src/eda.py
```
*Writes:* `results/eda/` — this is where the "LIAR is ~17 words, ISOT is ~380" figure comes from.

---

## Stage 2 — Generate *(the only steps that spend money)*

All four call the OpenAI API. **Flag this clearly in a review** — these are the
steps that cannot be re-run for free, which is why the generated data is
committed to the repository.

| Command | What it produces |
|---|---|
| `python src/generate_synthetic_fake.py --n 500` | 500 fake articles: real ISOT article, one fact altered |
| `python src/generate_synthetic_real.py --n 1000` | 1000 paraphrased real articles, facts unchanged — **the control for the authorship check** |
| `python src/generate_synthetic_fake_liar.py --n 200` | Same as the first, but sourced from LIAR — for the diverse-sourcing test |
| `python src/generate_style_attack.py --n_per_class 100` | Tone-only rewrites of held-out articles (the RQ4 attack) |
| `python src/generate_style_attack_reverse.py --n_per_class 100` | The opposite tone direction, to test whether the fix generalises |
| `python src/generate_counter_style_training.py --n_per_class 100` | Tone-shifted twins **for training** — the RQ4 fix itself |

> "Generation is non-deterministic — temperature 0.7 — so re-running produces
> different text and would not reproduce my numbers. That is exactly why
> `data/synthetic/` is version-controlled while everything else under `data/`
> is not."

---

## Stage 3 — Assemble the recipes

These do no machine learning. They combine real and synthetic rows into the
training sets, holding size constant so results reflect *composition*.

| Command | Builds |
|---|---|
| `python src/build_core_datasets.py` | `real_real`, `mixed`, `real_syn` — the three core recipes |
| `python src/build_augmented_datasets.py` | The augmentation variants |
| `python src/build_synthetic_real_datasets.py` | `c2_synreal_realfake`, `c3_synreal_synfake` — the authorship controls |
| `python src/build_swap_sweep_datasets.py` | The 0/25/50/75/100% synthetic sweep points |
| `python src/build_multisource_dataset.py` | `real_syn_multisource` — synthetic fakes from two sources |
| `python src/build_style_robust_dataset.py` | `style_robust` — baseline plus the tone-shifted twins |
| `python src/build_test_sets.py` | `test_indomain`, `test_crossdomain` (LIAR), `test_crossdomain2` (WELFake) |

**The one to explain properly is `build_test_sets.py`:**

> "This is where the evaluation design lives. The real-news side is held-out
> ISOT in every test set; only the fake side changes. So when I compare LIAR
> against WELFake I'm changing one variable, not two."

---

## Stage 4 — Train

**`train.py`** — one entry point for all four models and every recipe.

```bash
python src/train.py --model all                                   # all 4 models, 3 core recipes
python src/train.py --model bert --dataset style_robust           # one model, one recipe
python src/train.py --model all --dataset real_real --max-words 300
```

| Flag | Does |
|---|---|
| `--model` | `lr_svm`, `cnn`, `bert`, or `all` |
| `--dataset` | Any recipe name — whatever `train_<name>.csv` exists |
| `--seed` | Override the seed; results save under a separate label so repeats don't overwrite |
| `--max-words N` | Cap every document at N words for **all** models — the RQ3 fairness control |
| `--grad_accum` | Gradient accumulation, if BERT runs out of VRAM |

*Reads:* `train_<recipe>.csv` + `test_crossdomain.csv` · *Writes:* `models/` and `results/metrics_*.json`

> "This file replaced five near-identical scripts. They were the same
> train-and-evaluate loop copy-pasted per model and per dataset — two of them
> differed by a single string. Now the combination is an argument rather than a
> file."

**If asked why LR and SVM are trained together:** they share the TF-IDF
vectoriser, so fitting it once and using it for both is both faster and
guarantees they see identical features.

---

## Stage 5 — Evaluate

**`evaluate.py`** — one entry point with eight subcommands:

```bash
python src/evaluate.py master           # collect all metrics into one table + heatmap
python src/evaluate.py leakage          # train/test overlap + corpus independence
python src/evaluate.py cross-target --dataset welfake   # re-score checkpoints on WELFake
python src/evaluate.py hard-examples --model bert --comp real_syn
python src/evaluate.py length-sweep     # truncate the test set, isolate length from domain
python src/evaluate.py seed-summary     # mean ± SD across seeds
python src/evaluate.py error-analysis   # confusion matrices, vocabulary overlap
python src/evaluate.py case-studies     # concrete style-attack flips with article text
```

**The three worth naming in a review:**

- **`leakage`** — found that ISOT contains 23.7% duplicate fake articles, and
  that 63.8% of WELFake also exists in ISOT. Fails if overlap exceeds 2%.
- **`hard-examples`** — prints the articles a model got *most confidently
  wrong*. This is what produced the two misclassified Reuters articles behind
  the authorship finding.
- **`length-sweep`** — shortens the test articles to separate a length effect
  from a domain effect.

**The dedicated experiment runners:**

| Command | Answers |
|---|---|
| `python src/run_swap_sweep_experiment.py` | RQ2 — the 0→100% synthetic curve |
| `python src/run_multiseed_robustness.py --comps <names>` | RQ3 — 3 seeds per condition for CNN/BERT |
| `python src/eval_style_robustness.py` | RQ4 — flip rates, no retraining |
| `python src/run_train_extra_experiments.py` | LR/SVM on the augmentation variants |
| `python src/run_deep_extra_experiments.py --models cnn bert` | Same for the deep models |
| `python src/check_synthetic_quality.py [--judge 50]` | Data quality: diversity, fact verification, plausibility |

> "`eval_style_robustness.py` doesn't retrain anything — it loads the existing
> checkpoints and scores them on the attacked articles. That's why the RQ4
> comparison is fair: same model, two versions of the same text."

---

## Stage 6 — Report

| Command | Produces |
|---|---|
| `python src/export_detector_model.py` | `detector_model.js` — LR weights for the browser demo |
| `python src/build_report.py` | `results_report.html` |
| `python src/build_walkthrough.py` | `pipeline_walkthrough.ipynb` |
| `python src/verify_detector.py` | Checks the browser scorer against sklearn (needs Node) |

**This is the part to be proud of in a code review:**

> "The report is *generated*, not written. Every number is read from
> `results/` when the page is built, so the page cannot disagree with the data.
> It used to be hand-written HTML with the figures typed in, and they drifted
> out of sync more than once — so I removed the possibility rather than trying
> to be careful. Same reason the notebook is generated by a script."

---

## Questions you will get about the code

**"Why so many files instead of a few?"**
> "Each does one thing, so any stage can be re-run without touching the others,
> and a failure is localised. The grouping is visible from the filename
> prefixes. `DEFENSE_PREP.md` has the longer answer."

**"How do you know the models are compared fairly?"**
> "Three shared modules. `preprocessing.py` means all four see identically
> cleaned text, `metrics.py` means all four are scored identically, and
> `config.py` means the settings live in one place. Plus I added `--max-words`
> after realising the models don't naturally read the same amount of each
> article — that's the RQ3 fairness control."

**"Can I reproduce this?"**
> "The code and the synthetic data, yes — `requirements.txt` is pinned to exact
> versions and `requirements-lock.txt` has all 104 packages. Training is a few
> hours on a GPU. What you can't reproduce exactly is the generation step,
> because it's non-deterministic — which is why that data is committed rather
> than regenerated."

**"What happens if I run the scripts out of order?"**
> "They fail with a message naming the script you need to run first. Each one
> checks its inputs exist before doing any work."

**"Is any of it tested?"**
> "Not unit tests, no — that's a fair criticism. What it has instead are
> pipeline-level checks: `evaluate.py leakage` fails the build if train and
> test overlap, `verify_detector.py` checks the browser model against sklearn,
> and `export_detector_model.py` asserts the vectoriser settings it depends on.
> Those catch the failures that would actually invalidate results."

---

## If you only remember three sentences

1. **32 files, six stages, one direction** — each stage reads only what the
   previous one wrote.
2. **Five shared modules mean no experiment differs in the plumbing** —
   cleaning, metrics and seeding are shared code.
3. **The report and the notebook are generated from the results files**, so
   they cannot disagree with the data.
