
import json

import joblib
import numpy as np

import config as cfg

COMP = "real_real"   # the honest default: trained on genuine data, not synthetic


def main():
    vec = joblib.load(cfg.MODELS_DIR / f"tfidf_{COMP}.joblib")
    lr = joblib.load(cfg.MODELS_DIR / f"lr_{COMP}.joblib")

    # Guard the assumptions the JavaScript scorer hard-codes. If any of these
    # ever change in config/train.py, the browser would silently compute
    # different features from the ones the model was fitted on.
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
        # Reported cross-domain performance, so the UI can state what this
        # model actually achieves rather than implying it is authoritative.
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
