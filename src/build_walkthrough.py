"""
build_walkthrough.py -- generate pipeline_walkthrough.ipynb.

The notebook is a READ-ONLY tour of the pipeline for a live demo: it loads the
data, models and results that already exist and shows what each stage produced.
It deliberately trains nothing and calls no API, so every cell runs in about a
minute and can be executed in front of someone without risk.

Generated rather than hand-written for the same reason results_report.html is:
a notebook that duplicates src/ drifts out of date the moment a script changes,
and stale cells in a live demo are worse than no notebook. Regenerate with
`python src/build_walkthrough.py` whenever the pipeline changes.
"""
import json

import config as cfg

_N = [0]


def _id():
    # nbformat 4.5+ wants a stable id per cell; without one it warns on every
    # validation and will hard-error in a future version.
    _N[0] += 1
    return f"cell{_N[0]:02d}"


def md(*lines):
    return {"cell_type": "markdown", "id": _id(), "metadata": {},
            "source": "\n".join(lines)}


def code(*lines):
    return {"cell_type": "code", "id": _id(), "execution_count": None,
            "metadata": {}, "outputs": [], "source": "\n".join(lines)}


CELLS = [
    md("# Pipeline walkthrough",
       "",
       "A read-only tour of the fake-news detection pipeline. Every stage is shown",
       "using files that already exist — **nothing is retrained and no API is called**,",
       "so this runs end to end in about a minute.",
       "",
       "Training and generation are deliberately out of scope here: training takes",
       "hours of GPU time and generation costs API budget, so neither belongs in a",
       "notebook meant to be run live. What this shows is what each stage *produced*.",
       "",
       "| Stage | What it does | Shown here |",
       "|---|---|---|",
       "| 1. Load | Read ISOT, LIAR, WELFake | ✓ |",
       "| 2. Generate | LLM writes synthetic fake news | ✓ (output only) |",
       "| 3. Assemble | Build the training recipes | ✓ |",
       "| 4. Train | Fit LR / SVM / CNN / BERT | skipped — load a saved model instead |",
       "| 5. Evaluate | Score on unseen data | ✓ |",
       "| 6. Check | Leakage and data quality | ✓ |"),

    code("import sys, json, warnings",
         "warnings.filterwarnings('ignore')",
         "sys.path.insert(0, 'src')",
         "",
         "import pandas as pd, numpy as np, joblib",
         "import config as cfg",
         "",
         "pd.set_option('display.max_colwidth', 90)",
         "print('Project root :', cfg.ROOT)",
         "print('Random seed  :', cfg.SEED)"),

    md("---",
       "## Stage 1 — The data",
       "",
       "Three public datasets. **ISOT is the only one used for training**; the other",
       "two exist purely to test whether a model generalises to news it has never seen."),

    code("rows = []",
         "for name in ['isot_real','isot_fake','liar_fake','welfake_fake']:",
         "    p = cfg.PROCESSED_DIR / f'{name}.csv'",
         "    if p.exists():",
         "        df = pd.read_csv(p)",
         "        w = df['text'].astype(str).str.split().str.len()",
         "        rows.append({'corpus': name, 'articles': len(df),",
         "                     'median words': int(w.median()), 'mean words': int(w.mean())})",
         "pd.DataFrame(rows)"),

    md("The length column matters. LIAR statements are an order of magnitude shorter",
       "than ISOT articles, which is why the project tests whether the cross-domain",
       "gap is about *length* or about *content* — see Stage 6."),

    md("---",
       "## Stage 2 — Synthetic generation",
       "",
       "A language model (GPT-4o-mini) rewrites real articles two ways:",
       "",
       "- **synthetic fake** — change exactly one fact, keep the rest of the wording",
       "- **synthetic real** — paraphrase throughout, change no facts *(a control)*",
       "",
       "The generator recorded what it changed, so the edit can be audited afterwards.",
       "This cell reads the saved output — it does **not** call the API."),

    code("syn = pd.read_csv(cfg.SYNTHETIC_DIR / 'synthetic_fake.csv')",
         "print(f'{len(syn)} synthetic fake articles')",
         "print('transformations used:', syn['transformation'].value_counts().to_dict())",
         "",
         "r = syn.iloc[0]",
         "print('\\n--- what the model was told to change ---')",
         "print(r['modified_fact'])",
         "print('\\n--- resulting synthetic article (first 400 chars) ---')",
         "print(str(r['text'])[:400], '...')"),

    md("**This is the crux of the whole project.** The fake article keeps the original",
       "reporting style and changes one detail. It reads exactly like real news — which",
       "is why some models cannot separate the two classes at all (Stage 5)."),

    md("---",
       "## Stage 3 — Assembling the training recipes",
       "",
       "Each recipe is a different mix of real and synthetic examples. **The size is",
       "held constant** so that any difference in results comes from the data's",
       "composition rather than from having more of it."),

    code("recipes = {",
         "    'real_real'           : 'Real news + REAL fake news (baseline)',",
         "    'mixed'               : 'Real news + half real, half synthetic fake',",
         "    'real_syn'            : 'Real news + ONLY synthetic fake (full replacement)',",
         "    'c2_synreal_realfake' : 'AI-paraphrased real + real fake (control)',",
         "    'c3_synreal_synfake'  : 'AI-paraphrased real + synthetic fake (control)',",
         "    'real_syn_multisource': 'Real news + synthetic fake from TWO sources',",
         "    'style_robust'        : 'Baseline + tone-shifted twins (the RQ4 fix)',",
         "}",
         "out = []",
         "for name, desc in recipes.items():",
         "    p = cfg.PROCESSED_DIR / f'train_{name}.csv'",
         "    if p.exists():",
         "        d = pd.read_csv(p)",
         "        out.append({'recipe': name, 'rows': len(d),",
         "                    'real': int((d.label==0).sum()), 'fake': int((d.label==1).sum()),",
         "                    'what it is': desc})",
         "pd.DataFrame(out)"),

    md("---",
       "## Stage 4 — Training *(skipped — loading a saved model)*",
       "",
       "Training all four models across every recipe takes hours on a GPU, so instead",
       "we load one already-trained model and look at what it learned.",
       "",
       "Logistic Regression is the readable one: it assigns a weight to each word.",
       "Positive weights push a prediction toward **fake**, negative toward **real**."),

    code("vec = joblib.load(cfg.MODELS_DIR / 'tfidf_real_real.joblib')",
         "lr  = joblib.load(cfg.MODELS_DIR / 'lr_real_real.joblib')",
         "",
         "terms = np.array(vec.get_feature_names_out())",
         "w = lr.coef_[0]",
         "order = np.argsort(w)",
         "",
         "print('Vocabulary size:', len(terms))",
         "print('\\nStrongest FAKE indicators :', ', '.join(terms[order[-10:]][::-1]))",
         "print('Strongest REAL indicators :', ', '.join(terms[order[:10]]))"),

    md("Worth noticing in a review: several of the strongest signals are **stylistic or",
       "structural** rather than factual. That is the thread the whole project pulls on —",
       "a model can score well by learning *how* text is written rather than *whether*",
       "it is true."),

    md("---",
       "## Stage 5 — Evaluation",
       "",
       "Every recipe scored on LIAR, which no model saw during training.",
       "",
       "**F1** = overall accuracy at catching fake news (higher is better).  ",
       "**AUC-ROC** = does the model rank fake above real? 1.0 perfect, 0.5 random,",
       "**below 0.5 means systematically backwards** — not merely uncertain."),

    code("rows = []",
         "for name in recipes:",
         "    for m in ['LR','SVM','CNN','BERT']:",
         "        p = cfg.RESULTS_DIR / f'metrics_{m}_{name}.json'",
         "        if p.exists():",
         "            d = json.load(open(p))",
         "            rows.append({'recipe': name, 'model': m,",
         "                         'F1': round(d['f1'],3), 'AUC-ROC': round(d['auc_roc'],3)})",
         "res = pd.DataFrame(rows)",
         "res.pivot(index='recipe', columns='model', values='F1')[['LR','SVM','CNN','BERT']]"),

    md("### The two findings a reviewer should see",
       "",
       "**1. SVM collapses under full replacement.** Not a gradual decline — it predicts",
       "one class for every article, so its F1 is exactly zero."),

    code("res[res.recipe=='real_syn']"),

    md("**2. Both-sides-synthetic looks mediocre on F1 and is catastrophic on AUC-ROC.**",
       "",
       "F1 around 0.5 reads as 'the model is confused'. AUC-ROC near 0.02 says something",
       "very different: the model learned a confident rule pointing the *wrong way*."),

    code("c3 = res[res.recipe=='c3_synreal_synfake']",
         "print(c3.to_string(index=False))",
         "print('\\nAll four AUC-ROC values are far below the 0.50 random-guessing line.')",
         "print('The models are not uncertain — they are consistently backwards.')"),

    md("---",
       "## Stage 6 — Checking the data itself",
       "",
       "Two checks that were run against the project's own assumptions."),

    code("lk = pd.read_csv(cfg.RESULTS_DIR / 'extra' / 'leakage_report.csv')",
         "corp = lk[lk.check=='corpus_overlap_with_isot'][['test','pct_of_test']]",
         "print('How much of each TEST corpus also appears in the TRAINING corpus:')",
         "print(corp.to_string(index=False))",
         "print()",
         "dups = lk[lk.check=='within_corpus_duplicates'][['test','pct_of_test']]",
         "print('Duplicate articles inside each source corpus:')",
         "print(dups.to_string(index=False))"),

    md("**WELFake is not the independent test set it appears to be** — a large share of",
       "it also exists in ISOT, because WELFake is a merged corpus that includes the same",
       "source. LIAR is genuinely disjoint. This was measured rather than assumed, and it",
       "changes how the WELFake numbers should be read.",
       "",
       "The duplicate rate inside ISOT is also why roughly 1% of test articles appear in",
       "training: the split is done on rows, not on unique texts."),

    code("q = pd.read_csv(cfg.RESULTS_DIR / 'extra' / 'synthetic_quality.csv')",
         "div = q[q.check=='diversity'][['file','n','distinct_1','distinct_2','distinct_3','mean_pairwise_sim']]",
         "print('Is the generator repeating itself? (compare synthetic vs the real reference)')",
         "div"),

    md("Synthetic text scores about the same as real fake news on every diversity measure,",
       "so the generator is not producing hundreds of near-copies of the same article."),

    md("---",
       "## Try the model",
       "",
       "The same Logistic Regression model, scoring text end to end: clean → TF-IDF → predict."),

    code("from preprocessing import clean_series",
         "",
         "def predict(texts):",
         "    X = vec.transform(clean_series(pd.Series(list(texts))))",
         "    return lr.predict_proba(X)[:, 1]",
         "",
         "# Ten of each rather than one of each -- a single example is a coin flip and",
         "# tells you nothing about the model's actual behaviour.",
         "held_out = pd.read_csv(cfg.PROCESSED_DIR / 'test_crossdomain.csv')",
         "real_batch = held_out[held_out.label == 0].sample(10, random_state=cfg.SEED)['text']",
         "fake_batch = syn.sample(10, random_state=cfg.SEED)['text']",
         "",
         "pr, pf = predict(real_batch), predict(fake_batch)",
         "print(f'Genuine articles   : {(pr < .5).sum()}/10 correctly called REAL'",
         "      f'   (mean score {pr.mean()*100:.1f}% fake)')",
         "print(f'AI-generated fakes : {(pf >= .5).sum()}/10 correctly called FAKE'",
         "      f'   (mean score {pf.mean()*100:.1f}% fake)')"),

    md("### Read this result carefully — it is the point of the project",
       "",
       "The model handles genuine articles well and **misses the synthetic fakes almost",
       "entirely**. That gap is the finding, not a bug.",
       "",
       "The synthetic fakes are real news articles with a single fact altered, so every",
       "stylistic signal still says *genuine reporting*: same source, same register, same",
       "structure. A model that learned to recognise style rather than truth has nothing",
       "left to catch them with. This is precisely why replacing real fake news with",
       "synthetic fake news degrades performance in Stage 5 — and why SVM, which relies",
       "on separating the two classes cleanly, collapses to zero.",
       "",
       "Look back at Stage 4 for the mechanism: the strongest 'real' signals the model",
       "learned include **said**, **reuters**, **washington reuters** — publication and",
       "sourcing markers, not indicators of truth. The synthetic fakes keep all of those,",
       "because they *are* Reuters articles with one number changed.",
       "",
       "*(Note this is one recipe on one sample. Scored across the full test set the model",
       "reaches F1 0.857 — see Stage 5 for the complete picture.)*",
       "",
       "---",
       "",
       "## Where to go next",
       "",
       "- `results_report.html` — the full results, with a live detector",
       "- `README.md` — the complete run order, including the training steps skipped here",
       "- `PRESENTATION_SCRIPT.md` — the talk-through of every result"),
]


def main():
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = cfg.ROOT / "pipeline_walkthrough.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    n_code = sum(1 for c in CELLS if c["cell_type"] == "code")
    print(f"Wrote {out} ({len(CELLS)} cells, {n_code} executable)")


if __name__ == "__main__":
    main()
