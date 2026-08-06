# Project status — handoff for literature work and further updates

This file exists to pick this project back up without re-deriving context —
either for finishing the literature review, or for further system/code
changes. It is a snapshot, not documentation of the system itself (see
`README.md` for that) or defense prep (see `DEFENSE_PREP.md` for that).

## What's actually done (research + system side)

- All 4 proposal objectives, plus one validity check beyond the proposal,
  are fully investigated: every major finding is traced to a mechanism (not
  just reported as a number), verified across at least 2 of {model
  architecture, dataset, random seed}, and checked for the most obvious
  alternative explanation before being written up as fact.
- `src/` (26 files) is organized into 6 pipeline stages + a shared core, with
  no remaining duplicated logic — checked twice, most recently confirming
  the OpenAI-generation scripts' shared prompts/plumbing now live in
  `gen_common.py` rather than being copy-pasted six times.
- `results_report.html` is the live, shareable summary — 8 sections, 14
  charts, 6 reference tables, all internally consistent (every claim in the
  text has a chart or table backing it, checked deliberately after a few
  gaps were found and closed).
- `DEFENSE_PREP.md` has the findings ranked by strength, methodology
  justification, 7 named limitations, and prepared answers to the questions
  most likely to come up in a viva.

## Thesis chapter writing — status by subsection

| Subsection | Status |
|---|---|
| 4.1 Introduction | Written by you, not reviewed here |
| 4.2.1 RQ1 (full replacement + synthetic-real) | Drafted, table + 2 figures provided, inserted into your doc |
| 4.2.2 RQ2 (augmentation) | Drafted, table + figure provided, inserted. **Flow fix given but not yet confirmed applied**: move the "Where RQ1 tested..." paragraph to before the figure |
| 4.2.3 RQ3 (cross-domain generalization) | Drafted, tables + 2 figures provided. **Heading/intro paragraph was mismatched with content — fix given** (new heading, delete the "condition C1" paragraph, add a new framing paragraph) — confirm this was applied |
| 4.2.4 RQ4 (style robustness) | Drafted, 3 tables + figure provided. **Same flow fix as 4.2.2**: move the intro paragraph before the figure, and add a sentence explicitly pointing at Figure 4.4 |
| 5.3.1–5.3.4 Discussion + Limitations | Reviewed your draft, found and fixed 2 factual errors (a misattributed AUC number, an overstated "LR never degrades" claim) and one outdated claim (C3's AUC anomaly called "unexplained" when it's already explained in 4.2.1/5.2.1). Limitations fully rewritten using data that already exists — final version delivered, ready to paste in |
| 5.3.5 Literature alignment | **Not written** — blocked on you confirming what your cited papers actually argue (see below) |

### Known gap: pages 13–16 of your thesis PDF were never reviewed

The PDF you sent (`draft c5.pdf`) was only 12 of 16 pages per its own footer.
Tables 4.5, 4.6, and 4.7 are referenced by the text in what I saw but never
appeared — they may be on the missing pages, or may not exist yet. **Re-export
and send the full document** to get a complete check, including whatever
comes after 4.2.4 (synthesis, limitations tie-in, possibly a conclusion).

## Literature review — what's actually blocking it

You sent ~33 citations (author/year/DOI, no titles or abstracts). I only
have reliable knowledge of a handful of these from general training
knowledge (BERT, the Transformer paper, the LIAR dataset paper, a couple of
well-known fake-news-detection surveys) — I used those safely. For
everything else, **I will not attribute a specific claim to a paper I
haven't actually read**, since that risks putting a fabricated claim into
your thesis under a real citation.

8 `[CITATION]` placeholders are sitting in the RQ sections, plus the whole of
5.3.5. To unblock these, send me **one line per paper**: what it actually
argues or found, specifically for the papers most likely to be relevant:

- Synthetic/LLM-generated data as a training substitute (relevant to 4.2.1, 4.2.2)
- Stylometric or authorship-based detection limits (relevant to 4.2.1's synthetic-real finding — this is the single most important citation slot in the whole chapter)
- Cross-domain generalization in fake-news detection (relevant to 4.2.3)
- Dataset/format bias in NLP benchmarks (relevant to 4.2.3)
- Stylistic/sentiment-based adversarial attacks on text classifiers (relevant to 4.2.4 — if Tahmasebi et al. 2026 is what your project's own code comments already frame this section around, confirm its actual argument)
- Adversarial training / paired counterfactual data as a defence (relevant to 4.2.4)
- TF-IDF/linear-model robustness under domain shift, specifically whether any source reports the same LR-robust/SVM-fragile asymmetry found here (relevant to 5.3.5)

Once I have those, I can write 5.3.5 properly and fill in the 8 placeholders
instead of leaving them as gaps.

## System/code — optional future work, not required

None of these are needed for the thesis to be complete; they're what's left
if you want to genuinely extend the research rather than just write it up:

- No human or LLM-judge quality check on synthetic text plausibility.
- No statistical significance testing anywhere (point estimates only).
- Seed-repeat data exists for only 7 of ~15+ conditions (`results/extra/multiseed_results.csv`) — style-robust, the diverse-sourcing fix, and 3 of 5 sweep points were never multi-seeded.
- Everything uses one LLM (GPT-4o-mini) and two related generation strategies — no second model or fundamentally different generation approach has been tried.
- Sample sizes (500–1,000/class) are cost-bounded; scaling up means new OpenAI spend, same as every prior generation step in this project.
- `models/` is ~3GB (mostly BERT checkpoints), flagged repeatedly for cleanup, never actioned — zero research value, pure housekeeping, do whenever.

## File map

| File | Purpose |
|---|---|
| `README.md` | Pipeline documentation, setup, research question map |
| `DEFENSE_PREP.md` | Viva/defense prep — findings ranked, methodology justification, limitations, anticipated Q&A |
| `PROJECT_STATUS.md` | This file |
| `results_report.html` | Live shareable results summary |
| `pipeline_structure.svg` | Diagram embedded in README (was missing from disk, restored just now) |
| `figure_4_1_auc_roc.png` | RQ1 — F1/AUC-ROC by model × recipe, all 5 recipes |
| `figure_4_1b_authorship_auc.png` | RQ1/RQ3 — authorship-shortcut AUC-ROC, LIAR vs WELFake |
| `figure_4_2_sweep_f1.png` | RQ2 — augmentation sweep line chart |
| `figure_4_3_crossdomain.png` | RQ3 — all 6 recipes, LIAR vs WELFake, 6-panel |
| `figure_4_4_style_flip_rate.png` | RQ4 — style-attack flip rate, all 4 models |
| `src/` (26 files) | Full pipeline — see `README.md`'s pipeline diagram for the grouping |

## Immediate next steps, in order

1. Confirm the 4.2.2/4.2.3/4.2.4 flow and heading fixes were actually applied in your live document (I can't check this myself — Google Docs isn't accessible to me directly).
2. Send pages 13–16 of the thesis PDF for a complete review.
3. Send one-line summaries of the papers most relevant to the 8 citation gaps, prioritizing the stylometry/authorship-detection one.
4. Decide whether to pursue any of the optional system work, or treat it as named future work in the thesis as-is.
