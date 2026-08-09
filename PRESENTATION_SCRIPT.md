# System review — presentation script

Everything you need to present this project confidently. Read Part 0 first if
any of the terminology is shaky; it makes the rest make sense.

**How to use this:** Parts 1–4 are what you *say*, in order. Part 5 is what you
say when questioned. Part 6 is the number sheet — glance at it, don't read it out.

---

## Part 0 — The vocabulary, in plain English

Learn these six and you can follow any question your supervisor asks.

**TF-IDF** — a way of turning an article into numbers a computer can compare.
It counts how often each word appears, then down-weights words that appear in
*every* article (like "said") because they don't distinguish anything. The
result is a long list of numbers standing in for the article.

**The four models.** All four do the same job — read an article, output "real"
or "fake" — but in different ways:
- **Logistic Regression (LR)** — the simplest. Learns a weight for each word:
  positive weights push toward "fake", negative toward "real". Adds them up.
- **SVM** — also word-based, but instead of weighting words it tries to draw a
  single dividing line between the two classes. If the classes look alike,
  there's no good line to draw, and it breaks. Remember this — it explains your
  most dramatic result.
- **CNN** — a small neural network that looks for short phrase patterns
  (3–5 words at a time) rather than individual words.
- **BERT** — a large pre-trained language model. It already "knows" English
  before seeing your data; you fine-tune it on your task. Most powerful,
  least predictable.

**F1 score** — overall accuracy at catching fake news, 0 to 1, higher is
better. It balances "did you catch the fakes" against "did you wrongly accuse
real articles".

**AUC-ROC** — a different question: does the model *rank* fake above real?
1.0 = perfect, 0.5 = coin flip. **Below 0.5 means the model is systematically
backwards**, not merely confused. This distinction is the single most important
idea in your project — one of your findings is invisible in F1 and obvious in
AUC-ROC.

**Cross-domain** — training on one dataset and testing on a *different* one.
Harder and more honest than testing on held-out data from the same source,
because real deployment always means unfamiliar data.

**Random seed** — neural networks start from random numbers. The seed fixes
which random numbers. Change the seed, retrain identically, and you can get a
different result. This matters more than you'd expect — see RQ3.

---

## Part 1 — What the system is (2–3 minutes)

> "The problem I started from is that real fake news is slow and expensive to
> collect. If a language model could just write it, we'd have unlimited
> training data. My project tests whether that actually works — and the answer
> turned out to be more interesting than a yes or no."

**The setup, in one sentence:** train on ISOT, test on LIAR and WELFake — two
datasets no model ever sees during training.

**Three datasets:**
- **ISOT** — full-length news articles, real and fake. The training data.
- **LIAR** — short political statements. Test only.
- **WELFake** — full-length articles from mixed sources. Test only.

**The pipeline, six stages** — this is the system-review part, so walk through it:

1. **Load** — read the three public datasets into a common format.
2. **Generate** — call GPT-4o-mini to write synthetic fake news. Two methods:
   take a real article and change one fact; or paraphrase a real article
   keeping every fact true (that second one is a control — I'll come back to it).
3. **Assemble** — build the training sets. Each "recipe" is a different mix of
   real and synthetic. Same size every time, so any difference is about the
   data's *composition*, not its quantity.
4. **Train** — the same four models on every recipe, using identical settings,
   so differences come from the data rather than the setup.
5. **Evaluate** — score everything on the unseen test sets, plus targeted
   experiments for each research question.
6. **Report** — one command regenerates the results page from the raw output
   files.

> "The important design decision is stage 6. Every number on my results page is
> read from the raw results files when the page is built. Nothing is typed in
> by hand. Earlier it *was* hand-typed, and figures drifted out of sync with the
> data more than once — so I made that impossible rather than just being careful."

**If asked about code quality**, this is the strongest thing you can say:
26 files, organised into those six stages plus a shared core. One cleaning
function used by all four models, so preprocessing can't be the reason two
models differ. One training entry point, one evaluation entry point.

---

## Part 2 — The four research questions (8–10 minutes)

### RQ1 — Can AI-written fake news replace real fake news?

> "Five recipes, changing only where the fake examples come from."

| | LR | SVM | CNN | BERT |
|---|---|---|---|---|
| Real + Real (baseline) | 0.857 | 0.908 | 0.956 | 0.964 |
| Real + Synthetic (full swap) | 0.885 | **0.000** | 0.711 | 0.673 |

> "Swapping in AI-written fake news costs CNN and BERT some accuracy, and LR
> actually holds up. But SVM doesn't degrade — it goes to zero. It predicts
> 'real' for every single article."

**Why** (say this, it shows you understand your own system): "My synthetic fake
news is made by changing one fact and keeping the original wording. So the fake
examples look almost identical to the real ones. SVM works by drawing a
dividing line between the two classes — when the classes look the same, there's
no line to draw."

**The answer:** partly, and only if the real side stays genuine.

---

### RQ1's validity check — the finding to lead with

> "Then I got suspicious. If a model does well on AI-written fake news, is it
> detecting *falsehood*, or just detecting *AI writing*? Those are very
> different skills. So I built a test."

The test: make the **real** side AI-written too — paraphrase real articles,
keeping every fact true. Now both classes are machine-written.

| Both classes AI-written | LR | SVM | CNN | BERT |
|---|---|---|---|---|
| F1 | 0.498 | 0.607 | 0.137 | 0.557 |
| **AUC-ROC** | **0.064** | **0.101** | **0.027** | **0.021** |

> "Look at F1 — around 0.5, which you'd read as 'the model is confused.' Now
> look at AUC-ROC: 0.02 to 0.10. Below 0.5 means worse than random. The model
> wasn't confused at all. It had learned a confident rule pointing exactly the
> wrong way."

**Why it happened** — this is the part that shows genuine understanding:

> "The cause was in how I built the data. Every 'real' example had been
> AI-paraphrased, so its sentence structure changed completely. Every 'fake'
> example kept its original wording, because only one fact changed. So the
> model learned 'reworded text = real, original-sounding text = fake.' At test
> time, genuine articles are never reworded — so they looked fake to it."

**The evidence:** two genuine Reuters articles, flagged as fake at 94% and
99.8% confidence, by two completely different model types. And it holds on both
test datasets, so it isn't one dataset's quirk.

**Why this matters beyond my project:** a good score on AI-generated fake news
doesn't prove you've built a fake-news detector. You have to check.

---

### RQ2 — Does *adding* synthetic data help, rather than replacing?

> "Here I fixed the fake class at exactly 500 examples and only changed how
> many were synthetic. Same size every time, so any difference is data quality."

| Synthetic share | 0% | 25% | 50% | 75% | 100% |
|---|---|---|---|---|---|
| LR | 0.857 | 0.925 | **0.937** | 0.925 | 0.885 |
| SVM | 0.908 | **0.915** | 0.803 | 0.007 | 0.000 |
| CNN | **0.953** | 0.947 | 0.880 | 0.774 | 0.719 |
| BERT | 0.950 | **0.998** | 0.989 | 0.978 | 0.669 |

> "LR, SVM and BERT all improve at 25%. Past 50%, SVM collapses. CNN is the odd
> one out — it never benefits at any level."

**On CNN** (have this ready, it's the obvious question): "I looked at what CNN
actually got wrong. The same error at every level — real articles called fake,
never the reverse — and the same *kind* of article: opinion-led writing rather
than plain wire reporting. Its confidence on those mistakes fell from 96% to
63% as I added synthetic data. So it's gradually losing its grip on something
it already found hard, not learning a wrong rule the way SVM does."

**The answer:** yes, in moderation — roughly 25–50% — and not for every model.

---

### RQ3 — Are transformers more consistently robust?

> "'Consistent' means two different things, and the models rank in opposite
> orders on them, so I measured both."

- **Robustness** — how much does the score move when the *training data* changes?
- **Stability** — how much does it move when you just *run it again*?

| | Robustness (spread across 6 recipes) | Stability (spread across 3 seeds) |
|---|---|---|
| **BERT** | **0.443 — best** | **0.385 worst case — worst** |
| LR | 0.478 | 0.000 (deterministic) |
| CNN | 0.819 | 0.095 |
| SVM | 0.908 — worst | 0.000 (deterministic) |

> "BERT is the *least* affected by which recipe you train it on, and it reaches
> the highest scores. But run the identical setup again with a different random
> starting point and BERT moves more than any other model."

**The number to land:**

> "Under full replacement, BERT's three runs gave 0.002, 0.662 and 0.676. Same
> data, same settings — one run simply failed to train. So a single BERT number
> is the least trustworthy figure in my study, and that's why every CNN and
> BERT result I report comes with its three-seed range."

**The answer:** no — not uniformly. No family wins on both.

**Also mention the fairness check** (shows rigour): "The four models don't
naturally read the same amount of each article — the two simpler ones read the
whole thing, BERT stops at 512 tokens, CNN at 300. So I retrained everything
capped at the same length. Two of three recipes were unaffected. Under full
replacement the ranking inverted: LR lost 0.218 F1 and went from best to worst.
So LR's apparent robustness was substantially about reading more text."

---

### RQ4 — Can style-diverse training resist tone attacks?

> "I took 200 unseen articles and rewrote them to change only their tone — real
> articles made sensational, fake ones made calm. Every fact and label
> unchanged. The flip rate is how often a model that was right changes its mind."

| Recipe | LR | SVM | CNN | BERT |
|---|---|---|---|---|
| Real + Mixed | 17.9% | 15.5% | 18.1% | 10.5% |
| **Style-robust** | **3.1%** | **2.6%** | **0.5%** | **0.0%** |

> "The counterintuitive part: just adding generic synthetic data made models
> *more* vulnerable, not less. The fix was to pair every article with a
> tone-shifted twin under the *same* label — so tone stops predicting the
> answer. That drops flips to near zero, and costs nothing: accuracy on normal
> data stayed the same or improved."

**Pre-empt the obvious challenge:** "I also tested the opposite attack
direction, which the models were never trained against — real made calm, fake
made dramatic. Flip rates rose slightly, BERT from 0% to 1.5%, LR from 3.1% to
4.6%, but stayed far below the 10–18% baseline. So the fix generalises, just
not perfectly."

**One honesty point worth volunteering:** "Two cells show 0.0% under full
replacement, for SVM and CNN. That isn't robustness — those models already
predict one class for almost everything under that recipe, so there's almost
nothing correct left to flip."

---

## Part 3 — Two things I found by checking my own assumptions (2 minutes)

This section is what separates a good project from a competent one. Volunteer it.

> "Two findings came from testing my own assumptions rather than my models."

**One — my 'independent' test set wasn't independent.**

> "I'd been describing WELFake as an independent second dataset. I measured it:
> 63.8% of its fake articles are exact text matches for articles that also
> appear in ISOT, my training corpus. LIAR is 0%. WELFake is a merged dataset
> that happens to include the same source ISOT comes from. So my WELFake scores
> are closer to an in-domain test than a cross-domain one, and I say so."

**Two — my explanation for the LIAR gap was wrong.**

> "I'd assumed LIAR scored worse because its text is much shorter. I tested
> that directly: kept the domain fixed and shortened the WELFake articles
> instead. Performance went *up*, not down. So length isn't the explanation —
> the cross-domain gap is a genuine domain effect, which makes the problem more
> real, not less."

**Also mention the integrity checks:** "ISOT contains duplicate articles —
23.7% of its fake class. Because the split is done on rows rather than unique
texts, about 1% of test articles also appear in training. The pipeline now
checks this automatically and fails if it exceeds 2%."

---

## Part 4 — Limitations (say these before you're asked)

> "Four honest limits."

1. **One generator.** All synthetic text came from GPT-4o-mini using two related
   methods. I can't claim these findings hold for other language models.
2. **Small samples.** 500–1,000 per class, bounded by API cost.
3. **Reported results predate a reproducibility fix.** I found that GPU training
   wasn't fully deterministic and fixed it, but the numbers I'm reporting were
   produced before that. They're reproducible going forward, not retroactively.
4. **Quality was measured, not assumed** — but partly with a caveat. The
   synthetic text is as varied as real fake news, and ~98% of the intended fact
   changes verify where the evidence was saved. The plausibility rating used an
   LLM judge from the same model family that generated the text, so
   self-preference bias can't be ruled out.

---

## Part 5 — Questions you will be asked, and what to say

**"Why Logistic Regression? It's ancient."**
> "Because it's a control. If a simple word-counting model matches a
> transformer, that tells me the task is being solved by surface features
> rather than understanding — which is exactly what I found in places. It's
> also deterministic, so it's the only model whose numbers I can guarantee
> reproduce exactly."

**"Why only 500 examples?"**
> "Cost. Every synthetic article is a paid API call. I chose to spend the
> budget on more *conditions* rather than more rows — six training recipes and
> four models tested two ways is more informative than one recipe at scale."

**"Isn't 0.00 F1 just a bug?"**
> "I checked. SVM predicts 'real' for every article, so it never gets a fake
> right — F1 is genuinely zero. It reproduces on both test sets and it's
> deterministic, so it isn't a fluke run. The cause is that my synthetic fakes
> are lexically near-identical to real articles, which leaves SVM no boundary
> to draw."

**"How do you know your synthetic fake news is any good?"**
> "I measured it three ways. Diversity — it's as varied as real fake news on
> every measure I tested. Fact verification — the generator recorded what it
> changed, and about 98% of those edits verify where the full source was saved.
> And an LLM plausibility rating, which it scored well on, though with the
> caveat that the judge shares a model family with the generator."

**"Why didn't you use a bigger/newer model?"**
> "The question wasn't which model is best — it was whether synthetic data can
> substitute for real data. Four model families across three architecture types
> answers that better than one strong model would."

**"Can I try it?"**
> Open the "Try it" tab. "This runs the actual Logistic Regression model in the
> browser — the real weights, not a simulation. Try the AI-generated fake
> example." *It predicts REAL at 21.9%.* "That's my whole thesis in one click:
> the synthetic fake is stylistically identical to real news, so the detector
> misses it."

**"What would you do next?"**
> "Three things. A second language model, to test whether these findings are
> about synthetic data in general or about GPT-4o-mini specifically. Statistical
> significance testing — I report point estimates and seed ranges but no formal
> tests. And a human quality check on the synthetic text, to replace the LLM
> judge with something free of self-preference bias."

**If you don't know an answer:**
> "I don't have that measured. What I *do* have is [nearest thing you tested]."
> Never guess. Your supervisor will respect the boundary more than a bluff.

---

## Part 6 — Number sheet (glance, don't read aloud)

**Baseline, Real+Real, tested on LIAR (F1):** LR 0.857 · SVM 0.908 · CNN 0.956 · BERT 0.964

**Full replacement (F1):** LR 0.885 · SVM 0.000 · CNN 0.711 · BERT 0.673

**Both classes AI-written (AUC-ROC):** 0.064 / 0.101 / 0.027 / 0.021 — all far below 0.5

**Augmentation sweet spot:** 25–50% synthetic. SVM collapses past 50%, CNN never benefits.

**Seed instability:** BERT worst-case SD 0.385; its three runs under full
replacement were 0.002, 0.662, 0.676. CNN 0.095. LR/SVM exactly 0.

**Style attack flip rates:** Real+Mixed 10.5–18.1% → Style-robust 0.0–3.1%.
Reverse direction 1.5–4.6%.

**WELFake overlap with ISOT:** 63.8%. LIAR: 0.0%.

**ISOT duplicate rate (fake class):** 23.7%, causing ~1% train/test overlap.

---

## Final delivery notes

**Lead with the validity check** (RQ1's AI-writing finding). If you only get
ten minutes, that's the material worth spending them on — it's a methodological
warning, not just a result, and it's the thing a supervisor remembers.

**Volunteer the limitations.** Every one sounds like rigour when you raise it
and like a gap when someone else finds it. You measured all four, which is
unusual.

**When you say a number, say what it means.** Not "AUC was 0.027" but "AUC was
0.027 — below 0.5, so the model was confidently backwards."

**Slow down on the two "I was wrong" findings.** WELFake not being independent,
and length not explaining the LIAR gap. Supervisors are looking for whether you
can find fault in your own work. You did, twice, with measurements.
