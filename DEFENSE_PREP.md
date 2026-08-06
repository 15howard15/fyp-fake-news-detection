# Defense Prep: Findings, Justification, and Anticipated Questions

This document is not project documentation (see `README.md` for that) — it's
preparation for explaining and defending this work out loud: what the
strongest finding is, why the methods and research questions were chosen,
and the specific weaknesses you should raise yourself before an examiner
does. Every claim below cites the actual number and where it comes from.

## One-paragraph summary

This project tests whether LLM-generated synthetic fake news can replace or
augment scarce real fake-news data for training fake-news detectors, across
four model families (LR, SVM, CNN, BERT) and evaluated cross-domain (train on
ISOT, test on LIAR and WELFake). The headline result is nuanced, not a clean
yes/no: synthetic replacement works for some models and breaks others
completely, the *reason* it breaks is traceable to a specific, named
mechanism (not just "some models are worse"), and one validity check
(beyond the proposal's 4 objectives) reveals a real risk in how synthetic-data experiments like this can be
built — a model can appear to detect "fake" while actually detecting
"AI-authorship," and that risk is confirmed, not just theorized.

## What's the strongest finding, and why

**Lead with the authorship-shortcut inversion (the validity check, beyond the
4 proposal objectives), not Objective 1's headline replacement numbers.** Reasoning:

- It's not just an anomalous number — it has a **traced mechanism**. C3
  (both classes AI-authored) doesn't just fail, it's *confidently backwards*:
  AUC-ROC sits well below 0.5 (LR 0.064, SVM 0.101, CNN 0.027, BERT 0.021 on
  LIAR — see `results_report.html`'s authorship chart). Reading the actual
  misclassified articles explained why: the synthetic-real training data is
  paraphrased (different sentence structure throughout), while the
  synthetic-fake training data is near-verbatim (one fact changed, rest
  untouched). The model learned "unedited wire-copy phrasing = fake" because
  that's what it was shown — a mechanistic, falsifiable explanation, not
  speculation.
- It's **quadruple-verified**: two independent architectures (BERT, LR) show
  the identical misclassification pattern; it reproduces on WELFake, not just
  LIAR; it holds across 3 random seeds; and it even shows up **in-domain**
  (same-source test data, no cross-domain shift), ruling out "doesn't
  generalize" as the explanation.
- It functions as a **validity check on the entire thesis premise**. Without
  it, Objective 1's replacement results could be read as "the model detects AI writing style," not
  "the model detects fake content" — which would undercut the whole
  replacement argument.

The Objective 1 multisource fix is the best **actionable** finding
(diversifying the synthetic source repairs SVM's total collapse, AUC
0.500→0.976 on LIAR, and CNN's inverted ranking, AUC 0.054→0.998) — but the validity check is
the one that demonstrates research maturity, because it's where the results
were interrogated for a way they could be wrong, rather than accepted at
face value.

## Why these methods

- **LR → SVM → CNN → BERT is a capacity/inductive-bias gradient**, not an
  arbitrary list: bag-of-words linear (no context) → local n-gram patterns
  with pretrained semantic embeddings (some context) → full contextual
  transformer (maximal context). This is what actually lets Objective 4
  ("compare robustness across model families") be answered: does robustness
  scale with model sophistication? Answer, per the data: no — SVM and CNN
  break under full replacement and the validity check, not BERT; but BERT shows a distinct WELFake-specific
  calibration weakness under Objective 1's multisource composition. Capacity
  doesn't buy uniform robustness.
- **TF-IDF for LR/SVM** is the standard interpretable baseline — it's why
  SVM's `real_syn` collapse can be explained precisely: every wrong
  prediction sits at exactly 0.500 confidence, because near-duplicate
  high-dimensional sparse vectors give `LinearSVC` no margin to separate on.
- **Isotonic calibration instead of default sigmoid (Platt) for SVM** —
  verified necessary, not cosmetic: default calibration can invert the AUC
  ranking on near-perfectly-separable TF-IDF features.
- **GloVe (pretrained) embeddings for CNN**, not trained-from-scratch —
  controls for "the CNN didn't have enough data to learn good embeddings" as
  a confound.
- **BERT with warmup/decay/gradient clipping** — the standard stability
  recipe. Skipping it produces a training failure that looks like a real
  result: F1 jumped from 0.66 to 0.9987 on the same composition once fixed —
  a caught methodology bug, not a modeling choice.

## Why these research questions (and why two weren't in the original proposal)

- **Objective 1's replacement angle** directly operationalizes the core objective.
- **Objective 1's augmentation angle** is the more realistic deployment scenario — most
  real-world teams have *some* real fake news, not zero.
- **The validity check (authorship-shortcut)** was added, beyond the 4
  objectives, after the project's own earlier results showed an "unexplained"
  AUC anomaly — investigating rather than ignoring an anomaly is the point.
- **Objective 4 (style robustness)** tests a specific, literature-motivated failure
  mode (tone-as-shortcut), and was extended with a reverse-direction attack
  to check the fix isn't just overfit to the one attack it was built against.
- **Objective 1's multisource fix** exists because the original single-source
  generation didn't actually satisfy the proposal's own wording ("diverse
  real-world sources") — closing that gap turned out to produce one of the
  strongest results in the project.

## Key results at a glance

| Section | Headline result |
|---|---|
| Objective 1 — Full replacement | Works for LR/CNN/BERT (F1 0.67–0.89 vs 0.86–0.96 baseline); **SVM collapses completely** (F1 = 0.000, predicts "real" for every test article) |
| Validity check — Authorship shortcut | C3 (both classes AI-authored) is not just weak, it's **confidently inverted** (AUC-ROC 0.02–0.10, well below chance), traced to a paraphrase-vs-verbatim-edit asymmetry in the data construction |
| Objective 1 — Partial augmentation | Moderate blend (25–50% synthetic) helps LR/SVM/BERT; **CNN never benefits at any blend level** — traced to a specific, real, colorful/opinion-style article type it progressively loses confidence on |
| Objective 1 — Diverse sourcing | Diversifying the synthetic source repairs SVM (AUC 0.50→0.98) and CNN (AUC 0.05→1.00) on LIAR; **confirmed on WELFake too** (AUC improves for all 4 models on a domain neither generation step touched), though BERT's F1 specifically regresses on WELFake (0.71→0.30) despite better ranking — a threshold-calibration wrinkle, not a broken model |
| Objective 4 — Style robustness | Generic synthetic data (`mixed`) is *more* vulnerable to tone attacks than no synthetic data at all; the purpose-built fix (`style_robust`) drops flip rate to ~0%, and **generalizes to the reverse attack direction** (stays under 5% flip rate vs. 10–18% for the vulnerable recipe), though not quite as completely as the original direction |

## Limitations — name these yourself before an examiner does

1. **Single LLM, single prompt design for all generation.** Every synthetic
   example came from one model with one fact-table-then-edit strategy. The
   validity check's finding is real, but it's *conditional on this generation methodology* —
   state findings as such, not as universal claims about "synthetic data."
2. **C2/C3's own construction has a confound baked in, which is also the
   finding.** Real-side synthetic data is paraphrased; fake-side synthetic
   data is near-verbatim edited. That's authorship *plus* construction
   method entangled, not a clean isolation of authorship alone. A sharper
   design would use the *same* construction method for both. Naming this
   yourself shows you understand your own experiment's boundaries.
3. **No human judgment of synthetic-text quality or plausibility.** The
   quality filter is a length-ratio heuristic (0.4–2.5× word count), not a
   human or model judgment of whether the synthetic fake news actually reads
   as plausible. For a fake-news-generation thesis, this is a real gap.
4. **BERT's seed instability.** Reliable on 2 of 3 seeds, but fails
   catastrophically on the third (F1 as low as 0.002 on `real_syn`) even with
   stability fixes applied. Report distributions, not single runs, if asked
   which BERT number to trust.
5. **Generation scale is small (500–1,000 synthetic samples per batch),
   capped by API budget.** Untested whether findings hold at 10x scale — name
   this as future work, not an oversight to be caught.
6. **`test_crossdomain`'s genre/length confound.** LIAR is short statements
   (~11–17 words), ISOT is full articles (~250–380 words) — part of any
   "poor generalization" on LIAR is format shift, not pure content
   generalization failure. Partially disentangled via WELFake (full-length,
   independent source) and in-domain re-checks, but the LIAR-specific numbers
   still carry this caveat.
7. **Phase 3/5 scope expanded from the proposal's single pipeline into a
   3-way comparison matrix** (`real_real`/`mixed`/`real_syn`, not just
   "train on real+synthetic"). This is a stronger evaluation design (a
   baseline is needed to claim synthetic "works"), but it is a deviation from
   the literal spec — frame it as a deliberate strengthening in the
   methodology chapter, not something to leave implicit.

## Anticipated questions and prepared answers

**"26 files in `src/` — why not fewer?"**
File count went through two real passes, not zero: the genuinely duplicated
code (model classes copy-pasted across 8 training scripts, and the OpenAI
API-call/retry logic copy-pasted across 6 generation scripts) was
consolidated into `train.py`, `evaluate.py`, and `gen_common.py`. What's left
is deliberately one file per pipeline step — each `build_*`/`generate_*`
script assembles a genuinely different dataset variant with different
sampling logic and a different output schema, not the same logic
re-skinned. The test applied throughout: would a bug fix here need to be
copy-pasted to other files? If yes, merge; if no, merging just relocates the
same complexity into `--mode` flags inside fewer, larger, harder-to-review
files — worse for maintainability, not better. The pipeline-structure diagram
in `README.md` shows the actual grouping: 6 stages plus a shared core, not
26 unrelated scripts.

**"Why should I believe this isn't just you finding what you went looking
for?"**
Several of the strongest findings were things the data forced you to notice,
not things set out to prove: the validity check's AUC inversion, CNN's total non-benefit from
augmentation under Objective 1, the SVM calibration bug, and BERT's WELFake F1 regression
under Objective 1 were all unplanned — discovered while checking whether an
earlier result was trustworthy, not hypothesized in advance.

**"Is this synthetic-data problem general, or specific to your prompts?"**
Specific to this generation methodology (fact-table extraction + single-fact
edit) — stated as a scope limitation, not overclaimed as a universal property
of LLM-generated text.

**"Your C2/C3 control isn't clean — real and fake sides used different
construction methods."** Agreed, and that's explicitly named in the
limitations above — it's simultaneously the finding's mechanism and the
experiment's main design weakness. A cleaner follow-up would hold
construction method constant across both classes.

**"Why does full replacement break SVM but not LR, which is also linear?"**
Different loss objectives: SVM's margin-based decision rule needs *some*
separating margin between classes; near-duplicate synthetic-vs-real vectors
(from single-fact-edit generation) leave no margin to find, so it collapses
to the majority prediction. LR's cross-entropy objective degrades more
gracefully under the same near-duplicate feature vectors — it doesn't need a
margin, just a probability gradient.

**"Does the Objective 1 fix actually work, or did the model just see LIAR's
writing style during training?"** Checked directly: the LIAR-sourced
synthetic data is built from LIAR's *real*-labeled statements (disjoint from
the `liar_fake` rows used in testing) — no text-level leakage. Verified via a
third dataset (WELFake) neither generation step touched: AUC-ROC improves for
all 4 models there too, which is real evidence the fix is about construction
diversity, not a preview of the test domain.

**"What would you do with more time/budget?"** Scale synthetic generation
beyond 500–1,000 samples to test whether findings hold; add a human or
LLM-judge plausibility check on synthetic text; rebuild C2/C3 with matched
construction methods across both classes to cleanly isolate authorship
detection; extend the reverse-attack generalization check (Objective 4) to C2/C3 and
the multisource composition.
