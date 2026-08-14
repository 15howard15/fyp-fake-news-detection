# Presentation script — walking through `results_report.html`

Open the report full-screen and present from it. This follows it **tab by tab**,
and within each chart **left to right, top to bottom**, with numbered stops ①②③
so you always know where to point next.

For every number it says **whether it is good or bad**, because "0.79" means
nothing on its own — the judgement is the content.

Read Parts 0 and 0b tonight. The rest you hold and glance at.

---

## Part 0 — Six terms you must own

**TF-IDF** — turning an article into numbers. Count each word, then reduce the
weight of words appearing in *everything* (like "said") because they distinguish
nothing.

**The four models:**

| Model | How it decides |
|---|---|
| **LR** | A weight per word. Positive → "fake", negative → "real". Add them up. |
| **SVM** | Draws one dividing line between the classes. **If the classes look alike there's no line to draw** — this explains your most dramatic number. |
| **CNN** | Small neural network spotting phrase patterns, 3–5 words at a time. |
| **BERT** | Large pre-trained language model, fine-tuned on your task. Most powerful, least predictable. |

**F1** — overall accuracy at catching fake news. Higher is better.

**AUC-ROC** — a *different* question: does it rank fake above real?
**1.0 perfect · 0.5 coin flip · below 0.5 = systematically backwards.**

> **This distinction carries your project.** F1 asks "how often right?" AUC-ROC
> asks "which direction is it pointing?" A model can look mediocre on F1 and be
> catastrophically wrong on AUC-ROC. That is exactly what you found.

**Cross-domain** — train on one dataset, test on a different one.

**Random seed** — neural networks start from random numbers; the seed fixes
which. Same data, different seed → possibly a different result.

---

## Part 0b — What counts as "good" *(memorise this table)*

Every judgement in this script comes from here. If asked "is that good?", this
is your answer.

### F1 score

| Range | Verdict | Say |
|---|---|---|
| 0.95 – 1.00 | Excellent | "near-ceiling" |
| 0.85 – 0.95 | Good | "solid, deployable" |
| 0.70 – 0.85 | Usable but degraded | "it works, but it's lost something" |
| 0.50 – 0.70 | Weak | "barely better than guessing" |
| below 0.50 | Broken | "not functioning as a detector" |
| exactly 0.00 | **Catastrophic** | "it never catches a single fake" |

### AUC-ROC — *the important one*

| Range | Verdict | Say |
|---|---|---|
| 0.95 – 1.00 | Excellent ranking | "it separates the classes cleanly" |
| 0.85 – 0.95 | Good | |
| 0.70 – 0.85 | Fair | |
| ≈ 0.50 | **No signal** | "the same as flipping a coin" |
| **below 0.50** | **Backwards** | **"worse than random — it learned a confident rule pointing the wrong way"** |

> Below 0.5 is not "worse". It is *inverted*. If you flipped its every answer,
> it would be a good detector. That is a much stranger and more interesting
> failure than being merely inaccurate.

### Flip rate (RQ4) — **lower is better**

| Range | Verdict |
|---|---|
| 0 – 3% | Robust |
| 3 – 10% | Some vulnerability |
| **10%+** | **Clearly fooled by tone alone** |

### Seed spread — standard deviation, **lower is better**

| Range | Verdict |
|---|---|
| 0.000 | Deterministic — identical every run |
| below 0.05 | Stable — one run is trustworthy |
| 0.05 – 0.15 | Noticeable wobble |
| **above 0.15** | **Unreliable — a single number means little** |

---

## Opening — 45 seconds, before clicking

> "Real fake news is slow and expensive to collect, which limits how well
> detectors can be trained. If a language model could just *write* fake news, we
> would have unlimited training data. My project tests whether that works.
>
> Everything trains on ISOT — real news and real fake news — and tests on LIAR
> and WELFake, which no model ever sees during training. Four models, four
> research questions."

---

# TAB 1 — RQ1 · Replacement

## Chart 1: "Detection performance by method and training recipe"

Four groups on the x-axis: **LR, SVM, CNN, BERT**. Five coloured bars in each.
Walk it group by group.

**Set up first:**

> "Five training recipes. The only thing changing is where the fake examples come
> from. The first bar in each group is the baseline — real news and real fake
> news."

### ① LR group (leftmost) — 0.857 · 0.937 · 0.885 · 0.793 · 0.498

> "LR baseline is 0.857 — solid. Add synthetic on top and it *improves* to 0.937.
> Replace real fake news entirely and it's 0.885, still above baseline."

**Is that good?**

> "Yes, genuinely good. LR barely notices the swap. If this were the only model I
> tested I'd conclude synthetic data works fine."

### ② SVM group — 0.908 · 0.803 · **0.000** · 0.790 · 0.607

**Point at the missing third bar.**

> "SVM baseline is 0.908, the best of the two traditional models. Then under full
> replacement — nothing. Zero."

**Is that good? — the key interpretation:**

> "Zero F1 doesn't mean 'poor'. It means it never catches a single fake article.
> It predicts 'real' for all of them. This is total failure, not degradation.
>
> And it tells us something specific about my data. I make fake news by taking a
> real article and changing one fact, keeping the wording. So the two classes
> look almost identical. SVM works by drawing a dividing line — when the classes
> look the same, there is no line to draw."

### ③ CNN group — 0.957 · 0.875 · 0.711 · 0.733 · **0.137**

> "CNN baseline is 0.957, near-ceiling. Full replacement drops it to 0.711 —
> still usable but clearly degraded. Then the last bar, 0.137."

**Is that good?**

> "0.711 is a real cost but survivable. 0.137 is broken — that's the recipe where
> both classes are AI-written, which is the next chart."

### ④ BERT group — 0.964 · **0.998** · 0.673 · 0.640 · 0.557

> "BERT's baseline is 0.964, and with a moderate synthetic blend it reaches
> 0.998 — the highest number on the chart. But full replacement takes it to
> 0.673."

**Is that good?**

> "0.998 is excellent, but I'd add a caveat I'll come back to in RQ3 — that
> number is one run, and BERT is unstable. The honest reading is that it's the
> top of a range, not a dependable outcome."

### ⑤ Switch the metric — click **AUC-ROC**

> "Same data, different question. A dashed line appears at 0.50 — random
> guessing. Look at CNN under full replacement: 0.054."

**Is that good?**

> "It's below the line, which is the important part. Not 'bad' — *backwards*. F1
> showed 0.711 for that same cell, which reads as merely mediocre. AUC-ROC
> reveals the model is confidently ranking in the wrong direction. This is why I
> report both."

*(Click back to **F1 score**.)*

### Before the answer — flip the corpus switch

There is a **Tested on** row under the chart. Use it; don't wait to be asked.

> "Every number I just gave you was on LIAR. Let me switch to cleaned WELFake —
> full-length articles, every ISOT article stripped out."

**Point at Real + Synthetic, LR:**

> "On LIAR, LR under full replacement reads 0.943 AUC. On WELFake it reads
> 0.653. Same model, same weights — only the test corpus changed."

**Why? — have this ready:**

> "Because a word-counter alone scores 0.9999 on LIAR and 0.44 on WELFake.
> LIAR's fake statements average 16 words and the real articles average 367, so
> a model can look good there by noticing length. Where the two corpora
> disagree, WELFake is the one to believe."

### RQ1 answer

> "So: partly, and less than LIAR alone suggests. LR, CNN and BERT stay usable
> under replacement on LIAR; on the corpus where length can't be exploited, only
> the recipes that keep real fake news hold up. SVM breaks entirely either way.
> And all of it depends on the real-news side staying genuine — which is what I
> checked next."

---

## Chart 2: the validity check — **your strongest material**

**Scroll to "Validity check: detecting fake news, or detecting AI writing?"**

**Set up — say this slowly:**

> "Partway through I got suspicious. If a model scores well on AI-written fake
> news, is it detecting *falsehood*, or just detecting *AI writing*? Those are
> completely different skills, and only one is useful.
>
> So I built a control. I made the **real** side AI-written too — paraphrased
> real articles, every fact kept true. Now both classes are machine-written. If
> the model were learning about truth, this shouldn't matter much."

Two bars per model. **Blue = only the fake class is AI-written. Orange = both.**

### ① LR — blue 0.797, orange **0.064**

> "Blue 0.797 — fair. Orange 0.064."

### ② SVM — blue 0.809, orange **0.101**

### ③ CNN — blue 0.958, orange **0.027**

> "This one is the sharpest. Blue 0.958 is excellent ranking. Orange 0.027."

### ④ BERT — blue 0.772, orange **0.021**

**Now deliver the interpretation — the key sentence of the whole presentation:**

> "Every orange bar sits between 0.02 and 0.10. The random-guessing line is 0.50.
>
> If the model were *confused*, it would sit **at** 0.5. It's at 0.02. That means
> it learned a clear, confident rule — and the rule points the wrong way. If you
> flipped every one of its answers, you'd have a good detector."

**Then the cause — this shows you understand your own system:**

> "The cause was in how I built the data. Every 'real' example had been
> AI-paraphrased, so its sentence structure changed completely. Every 'fake'
> example kept its original wording, because only one fact changed. So the model
> learned: *reworded text means real, original-sounding text means fake.* At test
> time genuine articles are never reworded — so they looked fake to it."

**Click "Tested on WELFake".**

> "Same pattern on the second corpus. Not one dataset's quirk."

**Close with the general lesson:**

> "This goes beyond my project: a good score on AI-generated fake news does not
> prove you've built a fake-news detector. You have to check."

---

# TAB 2 — RQ2 · Augmentation

## The line chart — read it left to right

X-axis: **0% → 25% → 50% → 75% → 100%** synthetic.

**Set up:**

> "RQ1 asked about *replacing*. RQ2 asks the practical question — does *adding*
> synthetic data on top help? I fixed the fake class at exactly 500 examples and
> changed only what fraction is synthetic. Same size every time, so any
> difference is about composition, not quantity."

### ① Left edge, 0% — everyone starts high

> "LR 0.857, SVM 0.908, CNN 0.953, BERT 0.950. That's the all-real baseline."

### ② 25% — most lines rise

> "LR 0.857 → 0.925. BERT 0.950 → 0.998, its best result anywhere. SVM edges up
> to 0.915."

**Is that good?**

> "Yes — and this is the useful finding. A quarter synthetic *improves* three of
> four models. Synthetic data is adding something real fake news alone didn't
> provide."

### ③ 50% — the peak, then divergence

> "LR peaks here at 0.937. BERT holds at 0.989. But SVM has already fallen to
> 0.803 — below its own baseline."

**Is that good?**

> "Mixed. The sweet spot is roughly 25 to 50%. Past halfway SVM is already losing
> ground."

### ④ 75% — SVM falls off a cliff

> "SVM drops to 0.007. Effectively zero."

**Is that good?**

> "That's collapse, not decline. Same mechanism as RQ1 — once synthetic fakes
> dominate, the two classes look alike and SVM has no boundary to draw."

### ⑤ 100% — the right edge

> "LR 0.885, still above baseline. BERT falls to 0.669. CNN 0.719. SVM zero."

**Is that good?**

> "This is the replacement case from RQ1, and it confirms it: more synthetic data
> is definitely not better."

### ⑥ The green CNN line — trace it with your finger, it only goes down

> "CNN is the exception. 0.953 → 0.947 → 0.880 → 0.774 → 0.719. It never benefits
> at any level."

**The explanation (the note is on screen):**

> "I read what CNN actually got wrong at each level. Always the same direction —
> real articles called fake, never the reverse — and the same kind of article:
> opinion-led writing rather than plain wire reporting. Its confidence on those
> mistakes fell from 96% to 63%."

**Is that good or bad?**

> "It's a different failure from SVM's, and that distinction matters. SVM learned
> a wrong rule. CNN is gradually losing its grip on a category it already found
> hard. Same downward line, completely different cause."

## Before the answer — the corpus switch again, and why RQ2 is safe

If he raised the length problem in RQ1, he will expect it here. Get in first.

> "The same question applies to this sweep, so I checked it. Two things."

**One — length never moves inside this experiment:**

> "A word-counter scores between 0.48 and 0.51 on the training data at *every*
> blend level. Real fake news averages 378 words and my synthetic fake news
> averages 376 — they're the same length. So sliding the synthetic fraction from
> 0 to 100% doesn't change the length distribution at all. The thing that moves
> across this sweep is the fake class's *source*, not its length."

**Two — flip to In-domain (ISOT), where length scores 0.474:**

> "And the shape reproduces. Flat until 75%, collapse at 100% — on a corpus
> where word-counting is useless. So the trend is real."

**The one honest exception — volunteer it:**

> "The 100% endpoint is the exception. There the corpora disagree: LR reads
> 0.943 on LIAR but 0.638 in-domain, CNN 0.065 against 0.424. At full
> replacement the models lean on the length gap LIAR gives them. That endpoint
> is exactly the recipe I rebuilt under length control — it's in the framework
> tab."

**If he asks about the gaps in the WELFake line:**

> "LR, SVM and BERT have no checkpoints saved at 25% and 75% — the sweep script
> only persists CNN there. I left those as gaps rather than drawing a line
> through them, because a connected line would claim a measurement I don't have.
> CNN is complete, which is why it's the only unbroken WELFake line."

### RQ2 answer

> "Yes — augmentation genuinely helps, unlike replacement. But only in
> moderation, not for every architecture, and I've shown the trend holds on a
> corpus where length can't explain it."

---

# TAB 3 — RQ3 · Model families

The most conceptual tab. Slow down.

## Set up the two axes — point at the grey box

> "This asks whether transformers are more *consistently robust*. The problem is
> that 'consistent' means two different things, and the models rank in opposite
> orders on them. So I measured both.
>
> **Robustness** — does performance hold when the *training data* changes?
> **Stability** — does the *same setup* reproduce when you run it again?
>
> Both are spreads, so on both **a smaller number is better**. That's what the
> down arrows mean."

## The three cards — left to right

### ① Traditional ML (LR · SVM)

> "Peak 0.977. Robustness: LR 0.478, SVM 0.908 — red, the worst on the chart.
> Stability: 0.000, green."

**What that means:**

> "Perfectly reproducible — run them a hundred times, identical output. But the
> most sensitive to what you train them on. SVM swings from 0.00 to 0.91 purely
> depending on the recipe."

### ② Deep learning (CNN)

> "Peak 0.957 — red, the lowest ceiling. Robustness 0.819, middling. Stability
> 0.095."

**What that means:**

> "CNN never wins anything, but it never badly loses either. If you want
> predictable behaviour, it's the safest of the neural models."

### ③ Transformer (BERT)

> "Peak 0.9998 — green, the highest. Robustness 0.443 — green, the *best*.
> Stability 0.385 — red, the worst."

**The key line:**

> "BERT is the most robust to changing the data, *and* the least reproducible
> when re-run. No family holds green on all three. That's the finding."

### ④ The failure modes — as important as the numbers

> "Each fails differently, and how it fails matters as much as how often.
>
> SVM **fails loudly** — F1 exactly zero, impossible to miss.
> CNN **fails quietly** — AUC inverts to 0.054 while F1 still reads a plausible
> 0.711. If you only checked F1 you'd never notice.
> BERT **fails intermittently** — one run in three."

---

## Evidence 1 — spread across training recipes

Bar = mean across six recipes. Line = worst recipe to best.

### ① LR — mean 0.825, line 0.498 → 0.977
### ② SVM — mean 0.570, line **0.000 → 0.908** *(the longest line)*

> "SVM's line spans the whole chart. Its result depends almost entirely on what
> you trained it on."

### ③ CNN — mean 0.718, line 0.137 → 0.957
### ④ BERT — mean 0.805, line **0.557 → 1.000** *(the shortest)*

**Is that good?**

> "BERT's spread is 0.443, the narrowest. It's never terrible and never far from
> its best. That's a genuine strength, and it's the half of the answer people
> expect."

---

## Evidence 2 — spread across random seeds

**Set up carefully:**

> "This is the other half, and it's the one people find surprising.
>
> CNN and BERT start from random numbers. The seed fixes which ones. So I
> retrained every condition three times, at seeds 42, 1 and 2. LR and SVM aren't
> on this chart because they're deterministic — their spread is exactly zero by
> construction, and a bar with no line would be meaningless.
>
> Bar = mean of three runs. Line = lowest run to highest. **Short line, trust one
> run. Long line, don't.**"

Nine conditions, yellow = CNN, pink = BERT.

### ① Scan the yellow bars first — all short lines

> "CNN's lines are short everywhere. R+R is 0.966 with a range of 0.949 to
> 0.975 — that's a spread of 0.026. Run it again, you get roughly the same
> answer."

**Is that good?** → "Yes. That's what reproducible looks like."

### ② Now the pink bars — several very long lines

> "R+Mix: mean 0.857, but the runs were 0.580, 0.991 and 0.999.
> R+R: mean 0.826, runs 0.598, 0.929, 0.950."

### ③ **R+Syn — the one to stop on**

> "Mean 0.447, and the line stretches almost the full height. The three runs were
> **0.002, 0.662 and 0.676**. Same data, same settings — one run simply failed to
> learn the task at all."

**Is that good or bad?**

> "It's the worst number in my study, and it's not about accuracy — it's about
> trust. It means any single BERT figure I report is one draw from a wide
> distribution. Several of my strongest BERT results sit near the top of their
> range rather than in the middle."

### ④ Then the short pink bars — volunteer this

> "But look at R+Syn div and Style-rob. BERT there is 1.000 with a range of
> 0.999–1.000, and 0.958 with 0.906–0.989. Tight.
>
> So BERT isn't unstable everywhere. The instability is concentrated in the core
> replacement recipes — which means my RQ4 and diverse-sourcing conclusions don't
> inherit this problem."

### ⑤ Open the table underneath

> "This is why the table shows the observed range next to the ± figure. With
> three runs, mean ± standard deviation is a summary, not an interval — doing
> arithmetic on it would imply a range of 0.06 to 0.83 for that condition, and
> neither endpoint ever happened."

### RQ3 answer

> "So: no. Transformers are not uniformly more consistent. BERT is the most
> robust to changing the data and the least reproducible when re-run. No family
> wins both."

---

# TAB 4 — RQ4 · Style attacks

**Set up:**

> "The last question is a different vulnerability: can a detector be fooled just
> by changing an article's *tone*, without touching a single fact?
>
> I took 200 held-out articles and rewrote them — real articles made sensational,
> fake ones made calm. Facts and labels unchanged. The flip rate is how often a
> model that was originally *right* changes its answer. Down arrow: lower is
> better."

## Chart 1 — four bars per model, left to right

### ① Real + Real, the baseline

> "LR 11.5%, SVM 9.3%, CNN 1.0%, BERT 2.0%."

**Is that good?**

> "Mixed. CNN and BERT are genuinely robust at 1–2%. But LR at 11.5% means more
> than one in ten of its correct answers flips on tone alone. The baseline
> detector *is* fooled by tone."

### ② Real + Mixed — the tall bars, the surprising one

> "LR 17.9%, SVM 15.5%, CNN 18.1%, BERT 10.5%."

**Is that good?**

> "No — and this is the counterintuitive result. Adding generic synthetic data
> made every model *more* vulnerable, not less. All four are now above 10%, which
> is clearly fooled. Worse than using no synthetic data at all."

### ③ Real + Synthetic — get ahead of the two zeros

> "Ignore SVM and CNN's 0.0% here. Those models already predict one class for
> almost everything under this recipe, so there's almost nothing correct left to
> flip. A low flip rate only means something next to a working F1.
>
> LR's 59.1% is real though — that's the worst vulnerability anywhere in the
> study."

### ④ Style-robust — short bars, the point of the slide

> "LR 3.1%, SVM 2.6%, CNN 0.5%, BERT 0.0%."

**Say what was done:**

> "Here I paired every article with a tone-shifted twin under the *same* label.
> A dramatic version and a calm version both appear labelled 'real', and both
> appear labelled 'fake'. Tone stops predicting the answer."

**Is that good?**

> "Every model is now in the robust band, under 3%. BERT is at exactly zero —
> the tone rewrite fooled it on nothing it previously got right. That's a drop
> from 10–18% down to 0–3%."

## The two closing lines that make it defensible

> "It costs nothing: the table shows accuracy on ordinary, unattacked data stays
> the same or improves. It's not a trade-off.
>
> And on the second chart I tested the *opposite* attack direction, which the
> models were never trained against — real made calm, fake made dramatic. Flip
> rates rise a little: BERT 0 to 1.5%, CNN 0.5 to 2.0%, SVM 2.6 to 4.1%, LR 3.1
> to 4.6%. But all stay far below the 10–18% baseline. So the fix generalises,
> just not perfectly."

## If he cuts you off — the one-sentence answer

> "Yes, style-diverse training resists tone attacks — but the benefit comes from
> pairing both tones with both labels, not from synthetic data in general, which
> made things worse."

---

# TAB 5 — Evaluation framework

**Set up:**

> "Every question so far was answered by training on ISOT and testing on data no
> model had seen. That shared protocol — not a separate research question — binds
> all four together. This tab is how well it holds up."

### ① The chart — use the recipe buttons

> "Each model's score on LIAR beside its score on WELFake, for any recipe."

**Flick to Real + Real:**

> "WELFake scores higher than LIAR across the board. You'd read that as 'the
> models generalise better to WELFake'."

### ② The red note — your second-strongest moment

> "And here's something I found by checking my own assumption. I'd been
> describing WELFake as an independent second dataset. I measured it: **63.8% of
> its fake articles are exact text matches for articles that also appear in
> ISOT** — my training corpus. LIAR is 0.0%."

**Is that good or bad?**

> "Bad, and I say so rather than letting the number stand. WELFake is a merged
> dataset that happens to include the same source ISOT comes from. So my WELFake
> scores are closer to an in-domain test than a cross-domain one. That's why they
> look better — not because the models generalise well."

### ③ Contamination control — the recovery

This is the part that separates "I found a problem" from "I handled a problem."
Do not skip it, and do not rush ② to get here — ② is the setup.

> "Finding the contamination isn't enough on its own, so I removed it and
> re-scored everything."

**Point at the four tiles, left to right:**

> "63.3% of WELFake's whole fake pool is verbatim ISOT text. Filtering that out —
> plus WELFake's own internal duplicates — leaves 10,978 unique articles out of
> 37,106. I sampled from those to build a second test set, same size, same real
> articles, only the fake pool differs. Re-measured overlap: **0.0%**. Now it is
> genuinely independent."

**Point at the chart — the pink bars are contaminated, the cyan bars are clean:**

> "Same trained models, no retraining — a model's weights don't depend on which
> test set you score it on afterwards. So every difference you see is caused by
> the filtering and nothing else."

**Is that good or bad? — the answer to rehearse:**

> "Good, in three ways.
>
> One: nothing moves much. The largest single change across 40 scores is 0.130.
> My conclusions weren't propped up by the shared articles.
>
> Two: 36 of the 40 scores move **down**. So the contaminated set was flattering
> every model equally — inflating the numbers, not scrambling the ranking. That
> is the harmless kind of contamination, and I can now show it rather than hope
> it.
>
> Three, and this is the one I'd point at: **all five of the below-chance AUC
> scores get further below chance once the shared articles are gone.** Cleaning
> the data made my worst result worse."

**Why that third point matters — say it explicitly:**

> "The obvious objection to my headline finding — that some models rank fake
> news *backwards* — is 'that's probably just leakage.' If leakage caused it,
> removing the leaked articles would push those scores back up toward 0.5.
> They went the other way. So leakage is ruled out as the cause."

**Be precise here — leakage is one explanation, not the only one.** A second
control (next section) tests a different one, and *that* one does explain part
of it. If you claim more than "leakage is ruled out", you will be walked back.
The honest position, which is stronger than overclaiming:

> "This rules out leakage. It doesn't rule out everything — I tested a second
> explanation separately, and that one turned out to matter."

**If he asks why you kept the contaminated set at all:**

> "Because the difference between the two is the measurement. If I'd quietly
> swapped in the clean set, you'd have to take my word that the contamination
> didn't matter. This way you can check."

### ④ Length control — your strongest single result

If you only have time for one deep result, make it this one. It is the part
that shows you interrogated your own evaluation rather than just reporting it.

**Start with the first chart — the word-counter:**

> "Before comparing models, I asked what score you'd get on each test set using
> *nothing but document length*. Not a model — literally counting words."

**Point at the highlighted LIAR bar:**

> "On LIAR, **AUC 0.9999**. A word-counter almost perfectly separates my main
> cross-domain test set. LIAR statements average 16 words; the real articles
> average 367."

**Is that good or bad? — this is the moment:**

> "It's bad, and it's bad for *my own headline numbers*, because LIAR is the
> test set every model in this project was evaluated on. It means a high score
> there is not by itself evidence that a model learned anything about truth. It
> might just be measuring length. The WELFake sets sit at 0.44 to 0.46, so
> that problem is specific to LIAR."

**Now the second chart — what I did about it:**

> "So I rebuilt the recipe with the confound removed. Same manipulations, same
> pairing, same class balance — but the synthetic fakes are written at 25, 100
> and 400 words, and each real article is cut to match its own pair. Length
> now carries zero information about the label."

**Switch the selector to LIAR:**

> "On LIAR the scores move in *both* directions. LR falls from 0.943 to 0.690.
> CNN rises from 0.054 to 0.552."

**The interpretation — rehearse this until it's automatic:**

> "Those look like opposite results but they have the same cause. LR was riding
> the length gap forwards, CNN was riding it backwards. Neither was reading the
> facts. Take the cue away and both collapse toward the middle — toward what
> they actually know, which is much less than the original numbers suggested."

**Switch to WELFake (ISOT removed) — the control:**

> "The obvious objection is 'your new data is just worse — shorter text, less
> information.' So I tested on cleaned WELFake, where word-counting scores
> 0.44 and can't help. There the length-controlled recipe is **better** for
> three of the four models — **+0.138 AUC on average**. CNN goes from 0.376 to
> 0.768."

> "So it isn't a weaker model. It's a model that lost a shortcut and is now
> being measured on something real."

**Volunteer BERT — do not let him find it:**

> "BERT is the exception: it gets worse on both, −0.07 on LIAR and −0.11 on
> WELFake. I report that rather than averaging it away. Three of four is a
> result; four of four would be a claim I can't support."

**If he asks why you didn't just replace the old recipe:**

> "Because the comparison is the finding. One number on its own can't tell you
> whether length mattered. Keeping both is what makes it measurable — same
> reason I kept the contaminated WELFake set."

### ⑤ "Is the generated text any good?"

> "The only quality gate during generation was a length filter, which catches
> refusals but says nothing about whether the text is good. So I measured it
> afterwards."

**Point at the tiles:**

> "Diversity — my synthetic text scores 0.888 on distinct-3 against 0.871 for
> real fake news. Essentially identical, so the generator isn't producing
> hundreds of near-copies.
>
> Fact verification — about 98% of the recorded edits verify where the full
> source article was saved."

**Volunteer the caveat:**

> "The plausibility rating used an LLM judge from the same model family that
> generated the text, so self-preference bias can't be ruled out. I report it
> with that attached."

---

# TAB 6 — Try it *(the closer)*

**Click ▸ Try it.**

> "This runs the actual Logistic Regression model in your browser — the real
> trained weights, not a simulation."

**Click "AI-generated fake", then Analyse. It says REAL, ~22%.**

> "It says real. That's not a bug — that's my thesis in one click."

**Is that good or bad?**

> "It's a failure, and it's the expected one. This synthetic fake is a genuine
> news article with one fact altered. Every stylistic signal still says
> legitimate reporting — same source, same register, same structure. A model that
> learned style has nothing left to catch it with."

**Point at the term list.**

> "And it shows which terms moved the decision — words like 'reuters' and 'said'.
> Publication markers, not indicators of truth. The synthetic fake keeps all of
> them, because it *is* a Reuters article."

---

# Closing — 60 seconds

> "Four conclusions.
>
> One — synthetic fake news works as a supplement, not a replacement, and only up
> to about half the fake class.
>
> Two — a high score doesn't prove the model learned the right thing. Mine
> learned to detect AI writing and looked fine on F1.
>
> Three — a more powerful model isn't automatically more dependable. BERT has the
> highest ceiling and the least reliable floor.
>
> Four — robustness to manipulation has to be designed in deliberately. More data
> alone made it worse."

**Then limitations, unprompted:**

> "Four honest limits. All synthetic text came from one language model, so I
> can't claim these generalise to others. Sample sizes are 500 to 1,000 per
> class, bounded by API cost. The results I'm reporting predate a reproducibility
> fix I've since added, so they're reproducible going forward but not
> retroactively. And the quality check used an LLM judge sharing a model family
> with the generator."

---

# Questions, with answers

**"Why Logistic Regression? It's ancient."**
> "It's a control. If a simple word-counting model matches a transformer, the
> task is being solved by surface features rather than understanding — which is
> what I found in places. It's also the only model whose numbers reproduce
> exactly."

**"Why only 500 examples?"**
> "Cost — every synthetic article is a paid API call. I spent the budget on more
> *conditions* rather than more rows. Six recipes across four models on two
> corpora is more informative than one recipe at scale."

**"Isn't 0.00 F1 just a bug?"**
> "I checked. SVM predicts 'real' for every article, so it never gets a fake
> right — F1 is genuinely zero. It reproduces on both test sets and it's
> deterministic, so it isn't a fluke run."

**"How do you know your synthetic fake news is any good?"**
> "Three ways, all on the framework tab. Diversity — as varied as real fake news.
> Fact verification — about 98% of recorded edits verify where the evidence was
> saved. And an LLM plausibility rating, with the self-preference caveat stated."

**"Why is BERT so unstable?"**
> "Fine-tuning a large pre-trained model on 1,000 examples is known to be
> sensitive to initialisation. One of my three seeds failed to converge at all.
> It's why I report ranges rather than single numbers."

**"Is 0.9998 real, or overfitting?"**
> "It's on a test set the model never saw, so it isn't overfitting in the usual
> sense. But it's one run, and that condition has a wide seed range — so I'd
> describe it as the top of a range rather than a dependable result."

**"Could you have used a newer model?"**
> "The question wasn't which model is best — it was whether synthetic data can
> substitute for real data. Four families across three architecture types answers
> that better than one strong model would."

**If you don't know:**
> "I don't have that measured. What I do have is [nearest thing you tested]."
> Never guess.

---

# Delivery notes

**If you only get ten minutes:** RQ1's validity check → RQ3's Evidence 2 → the
Try-it demo. Those three carry the project.

**Always pair a number with its meaning.** Not "AUC was 0.027" but "AUC was
0.027 — below 0.5, so the model was confidently backwards."

**Slow down on the two places you proved yourself wrong** — the AI-writing
shortcut, and WELFake not being independent. Supervisors are looking for whether
you can find fault in your own work.

**Volunteer every limitation.** Each sounds like rigour when you raise it and
like a gap when someone else does.
