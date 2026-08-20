"""
export_detector_model.py -- ship the Logistic Regression detector to the browser.

The results page is a static file, so a "try it yourself" box has to run the
model client-side. That is only honest if it runs the ACTUAL trained model
rather than an approximation, which rules out pruning: the top 5,000 features
by coefficient magnitude carry just 42% of the total weight, so a trimmed model
would give different answers from the one the thesis reports.

LR is chosen over CNN/BERT for three reasons: it is a linear model, so scoring
is a dot product that reimplements exactly in JavaScript; it is deterministic,
so the browser's answer is reproducible; and its full weights are 1.45 MB,
which a page can load on demand. A transformer could not be shipped this way.

Everything needed to reproduce sklearn's pipeline goes into the bundle -- the
vocabulary, the IDF vector, the coefficients, the intercept, and the NLTK
stopword list -- because preprocessing.clean_text() strips stopwords before
vectorising and the browser must do the same or the features won't line up.

Writes detector_model.js next to results_report.html.
"""
import json

import joblib
import numpy as np

import config as cfg

COMP = "real_real"


def main():
    vec = joblib.load(cfg.MODELS_DIR / f"tfidf_{COMP}.joblib")
    lr = joblib.load(cfg.MODELS_DIR / f"lr_{COMP}.joblib")

    assert vec.ngram_range == (1, 2), vec.ngram_range
    assert vec.norm == "l2" and vec.use_idf and vec.smooth_idf
    assert not vec.sublinear_tf and vec.lowercase

    from nltk.corpus import stopwords
    sw = sorted(stopwords.words("english"))

    inv = {i: t for t, i in vec.vocabulary_.items()}
    terms = [inv[i] for i in range(len(inv))]

    bundle = {
        "comp": COMP,
        "terms": terms,
        "idf": [round(float(x), 5) for x in vec.idf_],
        "coef": [round(float(x), 6) for x in lr.coef_[0]],
        "intercept": float(lr.intercept_[0]),
        "stopwords": sw,
        "meta": {"f1": 0.857, "auc": 0.980, "trained_on": "ISOT (real news + real fake news)",
                 "tested_on": "LIAR (unseen)"},
    }

    out = cfg.ROOT / "detector_model.js"
    payload = json.dumps(bundle, separators=(",", ":"))
    out.write_text(f"window.DETECTOR_MODEL={payload};", encoding="utf-8")
    mb = out.stat().st_size / 1024 / 1024
    print(f"Wrote {out} ({mb:.2f} MB, {len(terms):,} features)")


if __name__ == "__main__":
    main()
