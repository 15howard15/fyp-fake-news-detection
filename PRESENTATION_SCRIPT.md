# Presentation script — walking through `results_report.html`

Open the report full-screen and present from it. This document follows it
**tab by tab, in order**. For every screen it tells you what to click, what the
audience sees, what to say, and — most importantly — **what the result
indicates**, because that is the part a supervisor is actually listening for.

Nothing here needs memorising. Read Part 0 tonight; the rest you can hold and
glance at.

---

## Part 0 — Six terms you must own

If any of these are shaky the rest won't land, because your best finding depends
on the difference between two of them.

**TF-IDF** — turning an article into numbers. Count how often each word appears,
then reduce the weight of words that appear in *everything* (like "said"),
because they don't distinguish anything. The article becomes a long list of
numbers.

**The four models** — same job, four ways of doing it:

| Model | How it decides |
|---|---|
| **Logistic Regression (LR)** | A weight per word. Positive pushes "fake", negative pushes "real". Add them up. |
| **SVM** | Draws one dividing line between the two classes. **If the classes look alike there is no line to draw** — remember this, it explains your most dramatic number. |
| **CNN** | A small neural network spotting short phrase patterns, 3–5 words at a time. |
| **BERT** | A large pre-trained language model that already knows English; you fine-tune it. Most powerful, least predictable. |

**F1 score** — overall accuracy at catching fake news. 0 to 1, higher is better.

**AUC-ROC** — a *different* question: does the model rank fake above real?
1.0 = perfect · 0.5 = coin flip · **below 0.5 = systematically backwards.**

> **This distinction carries your whole project.** F1 asks "how often right?"
> AUC-ROC asks "which direction is it pointing?" A model can look mediocre on
> F1 and be catastrophically, confidently wrong on AUC-ROC. That is exactly what
> you found.

**Cross-domain** — train on one dataset, test on a *different* one. Harder and
more honest than testing on held-out data from the same source.

**Random seed** — neural networks start from random numbers; the seed fixes
which ones. Same data, same settings, different seed → possibly a different
result.

---

## Opening (before you click anything) — 45 seconds

> "Real fake news is slow and expensive to collect, which limits how well
> detectors can be trained. If a language model could just *write* fake news, we
> would have unlimited training data. My project tests whether that actually
> works.
>
> Everything is trained on ISOT — real news articles and real fake news — and
> tested on LIAR and WELFake, which no model ever sees during training. Four
> models, four research questions."

Then open the report.

---

## TAB 1 — RQ1 · Replacement

### Screen 1: the main chart

**Click:** RQ1 tab. The chart is already on **F1 score**.

**What they see:** four groups (LR, SVM, CNN, BERT), five coloured bars each.

**Say:**

> "Five training recipes. The only thing changing is where the fake examples
> come from. Blue is the baseline — real news and real fake news."

**Point at SVM's third bar (0.00).**

> "Every recipe degrades a little when I swap in AI-written fake news — except
> SVM, which doesn't degrade at all. It goes to zero."

**What it indicates — say this:**

> "Zero F1 means it never catches a single fake article. It predicts 'real' for
> everything. And that tells us something specific about my synthetic data: I
> make fake news by taking a real article and changing one fact, keeping the
> wording. So the fake examples look almost identical to the real ones. SVM
> works by drawing a dividing line between two classes — when the classes look
> the same, there's no line to draw."

**The RQ1 answer:**

> "So the answer is: partly. LR, CNN and BERT stay usable. SVM breaks. And it
> only works at all if the real-news side stays genuine — which is what I
> checked next."

### Screen 2: switch the metric

**Click:** the **AUC-ROC** button above the chart.

> "Same data, different question. A dashed line appears at 0.50 — that's random
> guessing. Notice CNN under full replacement drops to 0.05, far *below* the
> line."

**What it indicates:**

> "Below 0.5 isn't 'bad'. It's backwards. The model is confidently ranking fake
> above real in the wrong direction. F1 hid that — it looked like 0.71, which
> reads as merely mediocre."

*(Click back to **F1 score** before moving on.)*

### Screen 3: the validity check — **your strongest material**

**Scroll to:** "Validity check: detecting fake news, or detecting AI writing?"

**Say — slowly, this is the centrepiece:**

> "Partway through I got suspicious. If a model scores well on AI-written fake
> news, is it detecting *falsehood*, or just detecting *AI writing*? Those are
> completely different skills, and only one is useful.
>
> So I built a control. I made the **real** side AI-written too — paraphrased
> real articles, every fact kept true. Now both classes are machine-written. If
> the model was learning about truth, this shouldn't matter much."

**Point at the orange bars: 0.06, 0.10, 0.03, 0.02.**

> "All four models, both test corpora, AUC-ROC between 0.02 and 0.10. Far below
> the 0.50 line."

**What it indicates — the key sentence of your whole presentation:**

> "If the model were confused, it would sit *at* 0.5. It's at 0.02. That means
> it learned a clear, confident rule — and the rule points the wrong way."

**Then explain the cause (this is what shows you understand your own system):**

> "The cause was in how I built the data. Every 'real' example had been
> AI-paraphrased, so its sentence structure changed completely. Every 'fake'
> example kept its original wording, because only one fact changed. So the model
> learned: *reworded text means real, original-sounding text means fake.* At test
> time, genuine articles are never reworded — so they looked fake to it."

**Click the "Tested on WELFake" button.**

> "Same pattern on the second corpus, so it isn't one dataset's quirk."

**Close with the general lesson:**

> "The takeaway goes beyond my project: a good score on AI-generated fake news
> does not prove you have built a fake-news detector. You have to check."

---

## TAB 2 — RQ2 · Augmentation

**Click:** RQ2 tab.

**Say:**

> "RQ1 asked about *replacing* real fake news. RQ2 asks the more practical
> question — does *adding* synthetic data on top help?
>
> I fixed the fake class at exactly 500 examples and changed only what fraction
> is synthetic. Same size every time, so any difference is about the data's
> composition, not about having more of it."

**Trace the lines with your finger, left to right:**

> "Most lines rise at 25%, then fall. LR peaks at 50%. BERT peaks at 25% and
> stays high until it collapses at 100%. SVM falls off a cliff after 50%."

**What it indicates:**

> "There's a sweet spot, roughly 25 to 50% synthetic. Below it you're leaving
> value on the table; above it you're diluting the real signal. And 'more
> synthetic data' is definitely not 'better'."

**Point at the green CNN line — it only ever goes down.**

> "CNN is the exception. It never benefits at any level."

**Then the explanation (the note is on screen):**

> "I read what CNN actually got wrong at each level. Always the same direction —
> real articles called fake, never the reverse — and the same kind of article:
> opinion-led writing rather than plain wire reporting. Its confidence on those
> mistakes fell from 96% to 63% as synthetic data increased."

**What that indicates:**

> "It's not learning a wrong rule like SVM did. It's gradually losing its grip
> on a category it already found hard. Different failure, different cause."

**The RQ2 answer:**

> "Yes — augmentation genuinely helps, unlike replacement. But only in
> moderation, and not for every architecture."

---

## TAB 3 — RQ3 · Model families

**Click:** RQ3 tab. This is the most conceptual tab — slow down.

### Set up the two axes first

**Point at the grey box at the top.**

> "This question asks whether transformers are more *consistently robust*. The
> problem is that 'consistent' means two different things, and the models rank
> in opposite orders on them. So I measured both.
>
> **Robustness** — does performance hold when the *training data* changes?
> **Stability** — does the *same setup* reproduce when you run it again?
>
> Both are measured as a spread, so on both of them a **smaller number is
> better** — that's what the down arrows mean."

### The three cards

> "Three families, three columns each. Green is best on that row, red is worst."

**Walk them across:**

> "BERT has the highest peak, 0.9998. BERT is also *best* on robustness — 0.443,
> the narrowest spread across recipes. But look at stability: 0.385, the worst of
> the three. The traditional models are the mirror image — exactly zero seed
> spread, because they're deterministic, but the widest swing across recipes."

**What it indicates:**

> "No family holds green on all three. That's the finding."

**Then the failure modes, which matter as much:**

> "Each also fails in a different *way*. SVM fails loudly — F1 exactly zero, you
> cannot miss it. CNN fails quietly — its AUC inverts to 0.054 while F1 still
> reads a plausible 0.711, so if you only check F1 you'd never notice. BERT fails
> intermittently — one run in three."

### Evidence 1

> "This backs the robustness column. The bar is each model's mean across six
> recipes; the line spans its worst to its best. SVM's line is enormous — it
> ranges from 0.00 to 0.91 depending purely on what it was trained on. BERT's is
> the shortest."

### Evidence 2 — **explain this carefully**

> "This backs the stability column, and it's the one people find surprising.
>
> CNN and BERT start from random numbers. The seed fixes which ones. So I
> retrained every condition three times, at seeds 42, 1 and 2. LR and SVM aren't
> here because they're deterministic — their spread is exactly zero by
> construction, and a bar with no line would be meaningless.
>
> The bar is the mean of three runs. The vertical line goes from the lowest run
> to the highest. **Short line means you can trust one run. Long line means you
> can't.**"

**Point at R+Syn, the pink bar with the enormous line.**

> "Under full replacement, BERT's three runs were 0.002, 0.662 and 0.676. Same
> data, same settings — one run simply failed to learn the task at all."

**What it indicates:**

> "It means any single BERT number in my study is one draw from a wide
> distribution, not a dependable outcome. Several of my strongest BERT results
> sit near the top of their range."

**Then the important qualifier — volunteer it:**

> "But notice the short lines on 'R+Syn div' and 'Style-rob'. BERT isn't
> unstable everywhere. The instability is concentrated in the core replacement
> recipes, which means my RQ4 and diverse-sourcing conclusions don't inherit
> this problem."

**Open the table underneath.**

> "This is why the table shows the observed range next to the ± figure. With
> three runs, mean ± standard deviation is a summary, not an interval —
> arithmetic on it would suggest a range of 0.06 to 0.83 for that condition,
> and neither endpoint ever happened."

**The RQ3 answer:**

> "So: no. Transformers are not uniformly more consistent. BERT is the most
> robust to changing the data and the least reproducible when re-run. No family
> wins both."

---

## TAB 4 — RQ4 · Style attacks

**Click:** RQ4 tab.

**Say:**

> "The last question is a different kind of vulnerability: can a detector be
> fooled just by changing an article's *tone*, without touching a single fact?
>
> I took 200 held-out articles and rewrote them — real articles made sensational,
> fake ones made calm and neutral. Facts and labels unchanged. The flip rate is
> how often a model that was originally *right* changes its answer. The down
> arrow means lower is better."

**Point at the tall bars for Real + Mixed.**

> "Here's the counterintuitive part. Just adding generic synthetic data made
> models *more* vulnerable — 10 to 18% of correct answers flip. That's worse than
> the plain baseline."

**Then the last group, Style-robust.**

> "The fix was to pair every article with a tone-shifted twin under the *same*
> label. So a dramatic version and a calm version both appear labelled 'real',
> and both appear labelled 'fake'. Tone stops predicting the answer."

**What it indicates:**

> "Flip rates drop to near zero — BERT to exactly 0%. And critically, it costs
> nothing: the table shows accuracy on normal, unattacked data stays the same or
> improves. It's not a trade-off."

**Volunteer the caveat before anyone asks — point at the two 0.0% cells under Real + Synthetic:**

> "Those two zeros aren't robustness. Under that recipe SVM and CNN already
> predict one class for almost everything, so there's almost nothing correct
> left to flip. A low flip rate only means something next to a working F1."

**Second chart:**

> "I also tested the *opposite* attack direction, which the models were never
> trained against. Flip rates rise a little — BERT from 0 to 1.5% — but stay far
> below the 10–18% baseline. So the fix generalises, just not perfectly."

**The RQ4 answer:**

> "Yes. And the benefit comes specifically from pairing both tones with both
> labels — not from synthetic data in general, which made things worse."

---

## TAB 5 — Evaluation framework

**Click:** Evaluation framework.

**Say:**

> "Every question so far was answered by training on ISOT and testing on data no
> model had seen. That shared protocol — not a separate research question — is
> what binds all four together. This tab is about how well it holds up."

**Use the recipe buttons to flick between two recipes.**

> "Each model's score on LIAR beside its score on WELFake, for any recipe."

**Then the red note — this is your second-strongest moment:**

> "And here's something I found by checking my own assumption. I had been
> describing WELFake as an independent second dataset. I measured it: **63.8% of
> its fake articles are exact text matches for articles that also appear in
> ISOT** — my training corpus. LIAR is 0%."

**What it indicates:**

> "WELFake is a merged dataset that happens to include the same source ISOT comes
> from. So my WELFake scores are closer to an in-domain test than a cross-domain
> one, and I say so rather than letting the number stand unqualified."

**Scroll to "Is the generated text any good?"**

> "The only quality gate during generation was a length filter, which catches
> refusals but says nothing about whether the text is any good. So I measured it
> afterwards. Diversity — my synthetic text is as varied as real fake news on
> every measure. Fact verification — the generator recorded what it changed, and
> about 98% of those edits verify where the full source was saved."

**Volunteer the caveat:**

> "The plausibility rating used an LLM judge from the same model family that
> generated the text, so self-preference bias can't be ruled out. I report it
> with that attached."

---

## TAB 6 — Try it *(the closer)*

**Click:** ▸ Try it.

> "This runs the actual Logistic Regression model in your browser — the real
> trained weights, not a simulation."

**Click the "AI-generated fake" example button, then Analyse.**

**Wait for the number. It says REAL, around 22%.**

> "It says real. That's not a bug — that's my thesis in one click."

**What it indicates:**

> "This synthetic fake is a genuine news article with one fact altered. Every
> stylistic signal still says legitimate reporting — same source, same register,
> same structure. A model that learned style has nothing left to catch it with."

**Point at the term list underneath.**

> "And it shows which terms moved the decision. Notice they're words like
> 'reuters' and 'said' — publication markers, not indicators of truth. The
> synthetic fake keeps all of them, because it *is* a Reuters article."

---

## Closing — 60 seconds

> "Four conclusions.
>
> One — synthetic fake news works as a supplement, not a replacement, and only up
> to about half the fake class.
>
> Two — a high score doesn't prove the model learned the right thing. Mine
> learned to detect AI writing and looked fine on F1.
>
> Three — a more powerful model isn't automatically a more dependable one. BERT
> has the highest ceiling and the least reliable floor.
>
> Four — robustness to manipulation has to be designed in deliberately. More data
> alone made it worse."

**Then the limitations, unprompted:**

> "Four honest limits. All synthetic text came from one language model, so I
> can't claim these findings generalise to others. Sample sizes are 500 to 1,000
> per class, bounded by API cost. The results I'm reporting predate a
> reproducibility fix I've since added, so they're reproducible going forward but
> not retroactively. And the quality check used an LLM judge that shares a model
> family with the generator."

---

## Questions, with answers

**"Why Logistic Regression? It's ancient."**
> "It's a control. If a simple word-counting model matches a transformer, the
> task is being solved by surface features rather than understanding — which is
> what I found in places. It's also the only model whose numbers I can guarantee
> reproduce exactly."

**"Why only 500 examples?"**
> "Cost — every synthetic article is a paid API call. I chose to spend the budget
> on more *conditions* rather than more rows. Six recipes across four models
> tested on two corpora is more informative than one recipe at scale."

**"Isn't 0.00 F1 just a bug?"**
> "I checked. SVM predicts 'real' for every article, so it never gets a fake
> right — F1 is genuinely zero. It reproduces on both test sets and it's
> deterministic, so it isn't a fluke run."

**"How do you know your synthetic fake news is any good?"**
> "Three ways, all on the framework tab. Diversity — as varied as real fake news.
> Fact verification — about 98% of recorded edits verify where the evidence was
> saved. And an LLM plausibility rating, with the self-preference caveat stated."

**"Why is BERT so unstable?"**
> "Fine-tuning a large pre-trained model on a small dataset — 1,000 examples — is
> known to be sensitive to initialisation. One of my three seeds failed to
> converge at all. It's why I report ranges rather than single numbers."

**"Could you have used a newer model?"**
> "The question wasn't which model is best — it was whether synthetic data can
> substitute for real data. Four families across three architecture types answers
> that better than one strong model would."

**If you don't know:**
> "I don't have that measured. What I do have is [nearest thing you tested]."
> Never guess.

---

## Delivery notes

**If you only get ten minutes:** RQ1's validity check, then RQ3's Evidence 2,
then the Try-it demo. Those three carry the project.

**Always say what a number means.** Not "AUC was 0.027" but "AUC was 0.027 —
below 0.5, so the model was confidently backwards."

**Slow down on the two places you proved yourself wrong** — the AI-writing
shortcut, and WELFake not being independent. Supervisors are looking for whether
you can find fault in your own work.

**Volunteer every limitation.** Each one sounds like rigour when you raise it and
like a gap when someone else does.
