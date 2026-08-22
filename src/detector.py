"""Export the Logistic Regression detector to the browser, and verify the port."""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import config as cfg
from preprocessing import clean_series

COMP = "real_real"
N = 25
TOL = 1e-6


def _load_model():
    """The vectorizer and classifier the browser copy has to reproduce."""
    vec = joblib.load(cfg.MODELS_DIR / "tfidf" / f"tfidf_{COMP}.joblib")
    lr = joblib.load(cfg.MODELS_DIR / "lr" / f"lr_{COMP}.joblib")
    return vec, lr


def cmd_export(args):
    """Dump the fitted vocabulary, IDF vector and coefficients to detector_model.js."""
    vec, lr = _load_model()

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
    return 0


def cmd_verify(args):
    """Score the same held-out articles in both runtimes and compare."""
    node = shutil.which("node")
    if not node:
        print("SKIPPED: node is not installed, so the JavaScript half cannot be run.")
        print("The scorer is therefore UNVERIFIED on this machine -- do not assume it matches.")
        return 2

    test = pd.read_csv(cfg.PROCESSED_DIR / "test_crossdomain.csv")
    sample = test.sample(n=N, random_state=cfg.SEED).reset_index(drop=True)

    vec, lr = _load_model()
    py = lr.predict_proba(vec.transform(clean_series(sample["text"])))[:, 1]

    root = cfg.ROOT
    harness = f"""
const fs=require('fs');
global.window={{}};
global.document={{createElement:()=>({{set innerHTML(v){{this._v=v;}},get value(){{
  return String(this._v).replace(/&amp;/g,'&').replace(/&lt;/g,'<').replace(/&gt;/g,'>')
    .replace(/&quot;/g,'"').replace(/&#39;/g,"'").replace(/&nbsp;/g,' ');}}}})}};
eval(fs.readFileSync({json.dumps(str(root / 'detector_model.js'))},'utf8'));
eval(fs.readFileSync({json.dumps(str(root / 'src' / 'detector.js'))},'utf8'));
const texts=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
console.log(JSON.stringify(texts.map(t=>{{const r=window.Detector.score(t);
  return r&&!r.empty?r.prob_fake:null;}})));
"""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "h.js").write_text(harness, encoding="utf-8")
        (td / "t.json").write_text(json.dumps(sample["text"].astype(str).tolist()), encoding="utf-8")
        res = subprocess.run([node, str(td / "h.js"), str(td / "t.json")],
                             capture_output=True, text=True, encoding="utf-8")
    if res.returncode != 0:
        print("FAIL: the JavaScript harness errored.\n", res.stderr[:1500])
        return 1

    js = np.array([np.nan if v is None else v for v in json.loads(res.stdout)])
    diff = np.abs(js - py)
    worst = int(np.nanargmax(diff))

    print(f"Compared {len(py)} held-out articles (composition: {COMP})")
    print(f"  max |JS - sklearn| = {np.nanmax(diff):.3e}")
    print(f"  mean               = {np.nanmean(diff):.3e}")
    print(f"  worst case: sklearn {py[worst]:.6f} vs JS {js[worst]:.6f}")
    agree = ((js >= .5) == (py >= .5))
    print(f"  same predicted label on {agree.sum()}/{len(py)}")

    if np.nanmax(diff) > TOL or not agree.all():
        print(f"\nFAIL: the browser scorer does not match sklearn within {TOL}.")
        return 1
    print(f"\nPASS: identical to sklearn within {TOL}.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("export", help="write detector_model.js from the trained LR model")
    sub.add_parser("verify", help="check the JS scorer matches sklearn (needs node)")
    args = ap.parse_args()
    return cmd_export(args) if args.command == "export" else cmd_verify(args)


if __name__ == "__main__":
    sys.exit(main())
