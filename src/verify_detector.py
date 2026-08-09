"""
verify_detector.py -- prove the browser scorer agrees with sklearn.

detector.js reimplements clean_text -> TfidfVectorizer -> LogisticRegression by
hand. A demo that quietly disagreed with the model it claims to be would be
worse than having no demo, so this runs both implementations over real held-out
articles and fails loudly if the predicted probabilities diverge.

The JavaScript is executed with Node if available. Without Node the check
cannot run, and it says so rather than passing silently.
"""
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


def main():
    node = shutil.which("node")
    if not node:
        print("SKIPPED: node is not installed, so the JavaScript half cannot be run.")
        print("The scorer is therefore UNVERIFIED on this machine -- do not assume it matches.")
        return 2

    test = pd.read_csv(cfg.PROCESSED_DIR / "test_crossdomain.csv")
    sample = test.sample(n=N, random_state=cfg.SEED).reset_index(drop=True)

    vec = joblib.load(cfg.MODELS_DIR / f"tfidf_{COMP}.joblib")
    lr = joblib.load(cfg.MODELS_DIR / f"lr_{COMP}.joblib")
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


if __name__ == "__main__":
    sys.exit(main())
