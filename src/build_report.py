"""
build_report.py -- regenerate results_report.html from the results files.

The report used to be hand-written HTML with every number typed into a
JavaScript literal. That works until a number changes: the tables and the
underlying results/*.json drift apart silently, and the only way to catch it
is to re-check every figure by hand (which is how several mismatches were
found). This script removes that failure mode -- every value in the report is
read from results/ at build time, so the report cannot disagree with the data
it claims to summarise.

Run it after any experiment that changes results/:

    python src/build_report.py

Layout: one tab per research question, an evaluation-framework tab for the
cross-domain protocol shared by all four, and a try-it tab running the LR model
client-side. Each chart carries a metric selector so accuracy / precision /
recall / F1 / AUC-ROC are all reachable rather than F1 alone.

matched_block() is retained but no longer emitted -- the matched-length
comparison was dropped from the report. The experiment itself still works
(`train.py --max-words 300`) and its metrics_*_max300.json files remain, so
re-adding the section is a one-line change to collect().
"""
import json

import pandas as pd
from sklearn.metrics import roc_auc_score

import config as cfg

RESULTS = cfg.RESULTS_DIR
EXTRA = RESULTS / "extra"
MODELS = ["LR", "SVM", "CNN", "BERT"]

RECIPE_LABEL = {
    "real_real": "Real + Real",
    "mixed": "Real + Mixed",
    "real_syn": "Real + Synthetic",
    "c2_synreal_realfake": "Synthetic-real + Real-fake",
    "c3_synreal_synfake": "Synthetic-real + Synthetic-fake",
    "real_syn_multisource": "Real + Synthetic (diverse-sourced)",
    "real_syn_mixedlen": "Real + Synthetic (length-controlled)",
    "c2_sym": "Synthetic-real + Real-fake (edit-matched)",
    "c3_sym": "Synthetic-real + Synthetic-fake (edit-matched)",
    "style_robust": "Style-robust",
}
METRICS = ["f1", "auc_roc", "accuracy", "precision", "recall"]


def _metrics_json(model, comp):
    p = RESULTS / f"metrics_{model}_{comp}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return {k: round(float(d[k]), 4) for k in METRICS if k in d}


def liar_block(comps):
    """Every model x composition on the LIAR cross-domain test set."""
    out = {}
    for comp in comps:
        row = {}
        for m in MODELS:
            v = _metrics_json(m, comp)
            if v:
                row[m] = v
        if row:
            out[comp] = row
    return out


def crosstarget_block(comps, fname="crossdomain2_results.csv"):
    """Read one of evaluate.py cross-target's per-target result files."""
    p = EXTRA / fname
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    out = {}
    for comp in comps:
        g = df[df.comp == comp]
        if g.empty:
            continue
        out[comp] = {
            r["model"]: {k: round(float(r[k]), 4) for k in METRICS if k in r}
            for _, r in g.iterrows()
        }
    return out


def welfake_block(comps):
    return crosstarget_block(comps)


def welfake_clean_block(comps):
    return crosstarget_block(comps, "crosstarget_welfake_clean_results.csv")


def contamination_block(comps):
    """WELFake scored twice: as shipped, and with every ISOT article removed.

    63.8% of the fake class in test_crossdomain2 is verbatim ISOT text, because
    WELFake is a merged corpus that absorbed the same Kaggle data ISOT derives
    from. That makes every "generalises to an independent long-form corpus"
    number on that set partly a re-test on training material. Rather than drop
    the contaminated set and quietly restate the numbers, both are reported:
    the gap between them is the measurement of how much the contamination was
    actually worth, and it is the only way a reader can check that the
    conclusions were not artifacts of it.

    Same checkpoints, same real class, same sample size -- only the fake-class
    pool differs, so the delta is attributable to the filtering and nothing
    else.
    """
    p = EXTRA / "crosstarget_welfake_clean_results.csv"
    q = EXTRA / "crossdomain2_results.csv"
    if not (p.exists() and q.exists()):
        return {}
    clean, dirty = pd.read_csv(p), pd.read_csv(q)
    out = {"comps": [], "rows": {}}
    for comp in comps:
        gc, gd = clean[clean.comp == comp], dirty[dirty.comp == comp]
        if gc.empty or gd.empty:
            continue
        row = {}
        for m in MODELS:
            rc, rd = gc[gc.model == m], gd[gd.model == m]
            if rc.empty or rd.empty:
                continue
            row[m] = {
                "f1_dirty": round(float(rd.iloc[0]["f1"]), 4),
                "f1_clean": round(float(rc.iloc[0]["f1"]), 4),
                "auc_dirty": round(float(rd.iloc[0]["auc_roc"]), 4),
                "auc_clean": round(float(rc.iloc[0]["auc_roc"]), 4),
            }
        if row:
            out["comps"].append(comp)
            out["rows"][comp] = row
    # Largest single move in either metric -- the headline "conclusions survive"
    # figure, computed rather than eyeballed so it cannot drift from the data.
    pairs = [(v[a], v[b])
             for r in out["rows"].values() for v in r.values()
             for a, b in (("f1_clean", "f1_dirty"), ("auc_clean", "auc_dirty"))]
    out["max_shift"] = round(max(abs(c - d) for c, d in pairs), 4) if pairs else None
    out["n_scores"] = len(pairs)
    out["n_down"] = sum(1 for c, d in pairs if c < d)
    out["n_up"] = sum(1 for c, d in pairs if c > d)
    # Does removing the shared articles rescue the below-chance AUCs, or make
    # them worse? If cleaning made them worse, the backwards-ranking result
    # cannot have been an artifact of ISOT text leaking into the test set --
    # which is the single most likely objection to that finding.
    bc = [(v["auc_clean"], v["auc_dirty"])
          for r in out["rows"].values() for v in r.values() if v["auc_dirty"] < 0.5]
    out["below_chance_total"] = len(bc)
    out["below_chance_worse"] = sum(1 for c, d in bc if c < d)
    # Recomputed from the corpora rather than copied from the build log, so the
    # figure in the report can never drift from the filter that produced the
    # test set. Same normalisation as build_test_sets.py.
    wf = cfg.PROCESSED_DIR / "welfake_fake.csv"
    if wf.exists():
        import re
        norm = lambda s: re.sub(r"\s+", " ", str(s)).strip().lower()
        isot = set()
        for stem in ("isot_real", "isot_fake"):
            isot |= set(pd.read_csv(cfg.PROCESSED_DIR / f"{stem}.csv")["text"].map(norm))
        keys = pd.read_csv(wf)["text"].map(norm)
        kept = keys[~keys.isin(isot)].nunique()
        out["pool_total"] = int(len(keys))
        out["pool_clean"] = int(kept)
        out["removed_pct"] = round(100.0 * (len(keys) - kept) / len(keys), 1)
        out["overlap_pct"] = round(100.0 * keys.isin(isot).mean(), 1)
    return out


SWEEP_PCT = {"swap_000": "0%", "swap_025": "25%", "swap_050": "50%",
             "swap_075": "75%", "swap_100": "100%"}
# The 0/50/100% points ARE real_real/mixed/real_syn -- build_swap_sweep_datasets
# reuses them by name rather than rebuilding identical files -- so their
# cross-target scores are stored under the composition name, not the sweep name.
SWEEP_ALIAS = {"swap_000": "real_real", "swap_050": "mixed", "swap_100": "real_syn"}

SWEEP_METRICS = ["f1", "auc_roc", "precision", "recall"]


def sweep_block():
    """RQ2 synthetic-fraction sweep, on every test set it has been scored against.

    Previously this kept only the cross-domain (LIAR) rows and discarded the
    rest. That mattered more than it looks: LIAR is separable at AUC 0.9999 by
    document length alone, so a sweep drawn only on LIAR cannot show whether its
    shape survives where word-counting doesn't help. The in-domain rows were
    already being computed and thrown away, and they are exactly that control.

    Three test sets, in increasing order of independence:
      - LIAR: the headline set, and the length-compromised one.
      - In-domain (ISOT): length-neutral (AUC 0.474 from length alone), but
        same-domain.
      - WELFake, ISOT removed: length-neutral AND a genuinely unseen corpus.
        Only partly available -- run_swap_sweep_experiment.py persists
        checkpoints for CNN at the 25%/75% points but not for LR/SVM/BERT, so
        those two columns are CNN-only. Missing cells stay null and render as
        gaps rather than being interpolated over.
    """
    df = pd.read_csv(EXTRA / "swap_sweep_results.csv")
    order = list(SWEEP_PCT.values())
    out = {"fractions": order, "tests": {}, "series": {}}

    LABEL = {"crossdomain": "LIAR", "indomain": "In-domain (ISOT)"}
    for test_key, label in LABEL.items():
        sub = df[df.test == test_key]
        if sub.empty:
            continue
        block = {}
        for m in MODELS:
            g = sub[sub.model == m]
            by = {SWEEP_PCT[r["sweep"]]: r for _, r in g.iterrows()
                  if r["sweep"] in SWEEP_PCT}
            block[m] = {met: [round(float(by[f][met]), 4)
                              if f in by and met in by[f] else None for f in order]
                        for met in SWEEP_METRICS}
        out["tests"][label] = block

    # Cleaned WELFake, assembled from the cross-target run rather than the sweep
    # script (which never scored against it).
    ct = EXTRA / "crosstarget_welfake_clean_results.csv"
    if ct.exists():
        cdf = pd.read_csv(ct)
        block, any_val = {}, False
        for m in MODELS:
            g = cdf[cdf.model == m]
            by = {}
            for sweep, pct in SWEEP_PCT.items():
                row = g[g.comp == SWEEP_ALIAS.get(sweep, sweep)]
                if not row.empty:
                    by[pct] = row.iloc[0]
            block[m] = {met: [round(float(by[f][met]), 4)
                              if f in by and met in by[f] else None for f in order]
                        for met in SWEEP_METRICS}
            any_val = any_val or bool(by)
        if any_val:
            out["tests"]["WELFake (ISOT removed)"] = block

    # How well document length alone separates the classes INSIDE each sweep
    # point's training data. Measured here rather than quoted, because it is the
    # claim the RQ2 note rests on: if this were to drift away from 0.5 at one end
    # of the sweep, the trend would be partly a length effect and the note would
    # be wrong. Real fake news and synthetic fake news happen to be the same
    # length (378 vs 376 words), which is why it stays flat -- that is a
    # property of the data, not something the design enforced, so it is checked.
    files = {"swap_000": "train_real_real", "swap_025": "train_swap_025",
             "swap_050": "train_mixed", "swap_075": "train_swap_075",
             "swap_100": "train_real_syn"}
    aucs = []
    for sweep, stem in files.items():
        p = cfg.PROCESSED_DIR / f"{stem}.csv"
        if not p.exists():
            continue
        t = pd.read_csv(p)
        if t["label"].nunique() < 2:
            continue
        w = t["text"].astype(str).str.split().str.len()
        aucs.append({"fraction": SWEEP_PCT[sweep],
                     "auc": round(float(roc_auc_score(t["label"], w)), 4)})
    if aucs:
        vals = [a["auc"] for a in aucs]
        out["trainLengthAuc"] = {"points": aucs,
                                 "min": round(min(vals), 3),
                                 "max": round(max(vals), 3)}

    # In-distribution CV across the same five points, both split strategies.
    # The gap between them widens with the synthetic fraction, which is the
    # cleanest available demonstration that the effect is caused by minimal
    # pairs: the 0% point has none and is unaffected.
    cv = cv_block([SWEEP_ALIAS.get(s, s) for s in SWEEP_PCT])
    if cv.get("comps"):
        by_frac = {}
        for sweep, pct in SWEEP_PCT.items():
            comp = SWEEP_ALIAS.get(sweep, sweep)
            if comp in cv["rows"]:
                by_frac[pct] = cv["rows"][comp]
        out["cv"] = {"fractions": [p for p in SWEEP_PCT.values() if p in by_frac],
                     "rows": by_frac, "splits": cv["splits"]}

    # Kept so anything still reading data.rq2.series keeps working.
    out["series"] = out["tests"].get("LIAR", {})
    return out


def style_block():
    cur = pd.read_csv(EXTRA / "style_robustness_results.csv")
    rev = pd.read_csv(EXTRA / "style_robustness_reverse_results.csv")
    recipes = ["real_real", "mixed", "real_syn", "style_robust"]
    flips = {RECIPE_LABEL[c]: {m: None for m in MODELS} for c in recipes}
    for _, r in cur.iterrows():
        if r["comp"] in recipes:
            flips[RECIPE_LABEL[r["comp"]]][r["model"]] = round(float(r["flip_rate"]), 4)
    reverse = {}
    for _, r in rev[rev.comp == "style_robust"].iterrows():
        reverse[r["model"]] = round(float(r["flip_rate"]), 4)
    original = {m: flips["Style-robust"][m] for m in MODELS}
    # Baseline (unattacked) performance, style_robust vs real_real
    baseline = {}
    for comp in ("real_real", "style_robust"):
        baseline[RECIPE_LABEL[comp]] = {m: _metrics_json(m, comp) for m in MODELS}
    return {"flips": flips, "original": original, "reverse": reverse,
            "baseline": baseline}


def length_block():
    p = EXTRA / "length_sweep_results.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    both = [t for t in ["full", "300w", "150w", "75w", "40w", "20w"]
            if t in set(df.truncated_to)]
    fake = [t for t in ["full", "300w/fake", "150w/fake", "75w/fake",
                        "40w/fake", "20w/fake"] if t in set(df.truncated_to)]
    def series(order):
        out = {}
        for (model, comp), g in df.groupby(["model", "comp"]):
            by = dict(zip(g.truncated_to, g.f1))
            key = f"{model} / {RECIPE_LABEL.get(comp, comp)}"
            out[key] = [round(float(by[t]), 4) if t in by else None for t in order]
        return out
    return {"both_labels": both, "both": series(both),
            "fake_labels": fake, "fake": series(fake)}


def matched_block(cap=300):
    """Full-text vs. matched-length results, read straight from the per-run
    metrics files.

    This deliberately does NOT read an intermediate CSV. The experiment is
    produced by `train.py --dataset <comps> --max-words 300`, which writes
    metrics_<MODEL>_<comp>_max<N>.json exactly like any other run, and those
    files are committed. Pairing them here means the whole chain -- command,
    per-run outputs, report -- is in the repository, with no hand-assembled
    file in between that nobody can regenerate.
    """
    out = {}
    for comp in ("real_real", "mixed", "real_syn"):
        row = {}
        for m in MODELS:
            full = _metrics_json(m, comp)
            cut = _metrics_json(m, f"{comp}_max{cap}")
            if full and cut:
                row[m] = {"f1_full": full["f1"], "f1_max300": cut["f1"],
                          "auc_full": full["auc_roc"], "auc_max300": cut["auc_roc"]}
        if row:
            out[RECIPE_LABEL.get(comp, comp)] = row
    return out


def families_block(comps):
    """RQ3 -- model-family consistency, which has two independent halves.

    "Consistent" can mean two different things and the models rank differently
    on each, so both are computed rather than collapsing them into one score:
      - spread ACROSS COMPOSITIONS: how much a family's score swings depending
        on what it was trained on (small = robust to the data recipe);
      - spread ACROSS SEEDS: how much the SAME setup moves between runs
        (small = a single reported number can be trusted). LR and SVM are
        deterministic given fixed data, so their seed spread is exactly zero
        by construction rather than by measurement -- which is itself part of
        the answer.
    """
    import statistics as st
    out = {"comps": comps, "byModel": {}}
    for m in MODELS:
        vals = []
        for c in comps:
            d = _metrics_json(m, c)
            if d:
                vals.append(d["f1"])
        if not vals:
            continue
        out["byModel"][m] = {
            "values": vals,
            "min": round(min(vals), 4), "max": round(max(vals), 4),
            "mean": round(sum(vals) / len(vals), 4),
            "range": round(max(vals) - min(vals), 4),
            "std": round(st.pstdev(vals), 4) if len(vals) > 1 else 0.0,
        }
    # seed spread, averaged over the conditions each model was re-run on
    p = EXTRA / "multiseed_results.csv"
    seed = {}
    if p.exists():
        df = pd.read_csv(p)
        df = df[df.test == "crossdomain"]
        g = df.groupby(["model", "comp"])["f1"].std().groupby("model")
        for model, s in g.mean().items():
            seed[model] = {"mean_std": round(float(s), 4),
                           "max_std": round(float(g.max()[model]), 4),
                           "deterministic": False}
    for m in ("LR", "SVM"):
        seed[m] = {"mean_std": 0.0, "max_std": 0.0, "deterministic": True}
    out["seedSpread"] = seed

    # Roll the four models up into the three families the objective actually
    # names, so the table answers the question as asked. LR and SVM belong
    # together because determinism -- the thing that makes them interesting
    # against BERT -- is a property of the family, not of either model alone.
    FAMILIES = [
        ("Traditional ML", ["LR", "SVM"]),
        ("Deep learning", ["CNN"]),
        ("Transformer", ["BERT"]),
    ]
    fam = []
    for label, members in FAMILIES:
        have = [m for m in members if m in out["byModel"]]
        if not have:
            continue
        peak = max(out["byModel"][m]["max"] for m in have)
        ranges = {m: out["byModel"][m]["range"] for m in have}
        seeds = {m: seed.get(m, {}) for m in have}
        deterministic = all(seed.get(m, {}).get("deterministic") for m in have)
        fam.append({
            "family": label,
            "members": have,
            "peak_f1": round(peak, 4),
            "recipe_range": {m: round(v, 4) for m, v in ranges.items()},
            "recipe_range_worst": round(max(ranges.values()), 4),
            "seed_std_worst": round(max((s.get("max_std", 0.0) for s in seeds.values()),
                                        default=0.0), 4),
            "deterministic": deterministic,
        })
    out["byFamily"] = fam
    return out


def seed_runs_block():
    """The individual per-seed F1 scores, not just their summary.

    mean +/- SD over three runs is a spread, not an interval: reconstructing
    endpoints from it produces numbers that were never observed (for BERT under
    full replacement it suggests roughly 0.06-0.83, where the actual runs were
    0.002, 0.662 and 0.676). Carrying the raw runs through to the report means
    the observed range can always be shown next to the SD.
    """
    p = EXTRA / "multiseed_results.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    df = df[df.test == "crossdomain"]
    out = {}
    for (model, comp), g in df.groupby(["model", "comp"]):
        out.setdefault(RECIPE_LABEL.get(comp, comp), {})[model] = \
            [round(float(v), 4) for v in sorted(g.sort_values("seed")["f1"])]
    return out


def seed_block():
    p = EXTRA / "multiseed_results.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    df = df[df.test == "crossdomain"]
    out = {}
    for (model, comp), g in df.groupby(["model", "comp"]):
        out.setdefault(RECIPE_LABEL.get(comp, comp), {})[model] = {
            "mean": round(float(g.f1.mean()), 4),
            "std": round(float(g.f1.std()), 4),
            "min": round(float(g.f1.min()), 4),
            "max": round(float(g.f1.max()), 4),
            "n": int(len(g)),
        }
    return out


def quality_block():
    p = EXTRA / "synthetic_quality.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    out = {"diversity": [], "fact": [], "judge": [], "truncation": []}
    for _, r in df.iterrows():
        c = r["check"]
        if c == "diversity":
            out["diversity"].append({
                "file": r["file"], "n": int(r["n"]),
                "d1": r["distinct_1"], "d2": r["distinct_2"],
                "d3": r["distinct_3"], "sim": r["mean_pairwise_sim"]})
        elif c == "fact_change":
            out["fact"].append({
                "file": r["file"], "n": int(r["n"]),
                "verified": r.get("edit_verified_pct"),
                "new_in_gen": r.get("new_in_generated_fuzzy_pct"),
                "novel": r.get("new_not_in_source_pct")})
        elif c == "fact_change_truncation_split":
            out["truncation"].append({
                "file": r["file"],
                "full": r.get("traceable_full_source_pct"),
                "trunc": r.get("traceable_truncated_source_pct")})
        elif c == "llm_judge":
            out["judge"].append({
                "file": r["file"], "n": int(r["n"]),
                "mean": r["mean_plausibility"], "pct45": r["pct_4_or_5"]})
    return out


def editsym_block():
    """The authorship-shortcut controls before and after the edit asymmetry was
    removed.

    C2/C3 pair synthetic-real against real-fake and synthetic-fake. Their fake
    class was edited far more lightly than their real class -- measured at 65.9%
    of the source retained against 44.0% -- so "close to the source wording"
    predicted "fake" without reference to any fact. C2'/C3' are the same
    controls built from a fake class rewritten to the same depth as the real
    class, and nothing else changed: same real rows, same manipulation
    strategies, same counts.

    Reported as a pair rather than a replacement. C3's below-chance AUC is a
    finding in its own right; what makes it interpretable is what happens to it
    when the asymmetry is removed.
    """
    pairs = [("c2_synreal_realfake", "c2_sym"), ("c3_synreal_synfake", "c3_sym")]
    if not _metrics_json("LR", "c3_sym"):
        return {}
    out = {"models": MODELS, "pairs": [], "tests": {}}
    for before, after in pairs:
        if _metrics_json("LR", after):
            out["pairs"].append({"before": before, "after": after})
    if not out["pairs"]:
        return {}

    comps = [c for p in out["pairs"] for c in (p["before"], p["after"])]
    out["tests"]["LIAR"] = liar_block(comps)
    clean = crosstarget_block(comps, "crosstarget_welfake_clean_results.csv")
    if len(clean) == len(comps):
        out["tests"]["WELFake (ISOT removed)"] = clean

    # The measured edit gap each side of the fix, recomputed here so the number
    # in the prose cannot drift from the corpora it describes.
    import difflib
    def sim(stem):
        p = cfg.SYNTHETIC_DIR / f"{stem}.csv"
        if not p.exists():
            return None
        df = pd.read_csv(p)
        if not {"text", "source_text"} <= set(df.columns):
            return None
        vals = [difflib.SequenceMatcher(None, str(s)[:4000], str(t),
                                        autojunk=False).ratio()
                for s, t in zip(df["source_text"], df["text"])]
        return round(sum(vals) / len(vals), 4)
    r, f_old, f_new = sim("synthetic_real"), sim("synthetic_fake"), sim("synthetic_fake_sym")
    if None not in (r, f_old, f_new):
        out["edit"] = {"real": r, "fake_before": f_old, "fake_after": f_new,
                       "gap_before": round(abs(r - f_old) * 100, 1),
                       "gap_after": round(abs(r - f_new) * 100, 1)}
    return out


def cv_block(comps):
    """5-fold CV for LR/SVM, under both an ordinary and a pair-aware split.

    Two things have to be said about these numbers or they will be misread:

    1. They are IN-DISTRIBUTION. Each fold trains on 80% of a composition and
       scores the held-out 20% of the same composition, where every other score
       in this report is cross-domain. A CV AUC of 0.99 next to a cross-domain
       AUC of 0.55 is not a contradiction, it is the gap between "separates data
       like its own" and "transfers".
    2. The ordinary split is not trustworthy on the synthetic recipes, and the
       grouped one is. The synthetic compositions are minimal pairs -- an article
       and its one-fact-altered twin -- so an ordinary split puts one half of a
       pair in train and the other in validation. The model memorises the
       article as real and calls its fake twin real too, which drives AUC below
       0.5 rather than merely lowering it. Keeping pairs whole removes that.
    """
    p = EXTRA / "cv_results.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    if "split" not in df.columns:
        return {}
    out = {"comps": [], "models": ["LR", "SVM"], "splits": [], "rows": {}}
    for s in ("stratified", "grouped"):
        if s in set(df["split"]):
            out["splits"].append(s)
    for comp in comps:
        g = df[df.comp == comp]
        if g.empty:
            continue
        block = {}
        for model in out["models"]:
            per_split = {}
            for s in out["splits"]:
                sel = g[(g.model == model) & (g["split"] == s)]
                if sel.empty:
                    continue
                per_split[s] = {
                    met: {"mean": round(float(sel[met].mean()), 4),
                          "sd": round(float(sel[met].std(ddof=1)), 4)}
                    for met in ("f1", "auc_roc", "accuracy")
                }
                per_split[s]["n_folds"] = int(len(sel))
            if per_split:
                block[model] = per_split
        if block:
            out["comps"].append(comp)
            out["rows"][comp] = block

    # Largest gap between the two splits, computed rather than quoted -- this is
    # the size of the near-duplicate effect and the reason the grouped split is
    # the one to report.
    gaps = []
    for comp, block in out["rows"].items():
        for model, per in block.items():
            if "stratified" in per and "grouped" in per:
                gaps.append({"comp": comp, "model": model,
                             "delta": round(per["grouped"]["auc_roc"]["mean"]
                                            - per["stratified"]["auc_roc"]["mean"], 4)})
    if gaps:
        out["gaps"] = sorted(gaps, key=lambda r: -abs(r["delta"]))
        out["max_gap"] = out["gaps"][0]
    return out


def significance_block():
    """McNemar's test results, from evaluate.py significance."""
    p = cfg.RESULTS_DIR / "statistical_significance.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    out = {"datasets": sorted(df["dataset"].unique().tolist()), "rows": []}
    for _, r in df.iterrows():
        out["rows"].append({
            "dataset": r["dataset"], "axis": r["axis"],
            "a": r["comparison_a"], "b": r["comparison_b"],
            "model": r["model"], "held": r.get("held_constant"),
            "n": int(r["n_samples"]), "n_disc": int(r["n_discordant"]),
            "acc_a": r["accuracy_a"], "acc_b": r["accuracy_b"],
            "p": float(r["p_value"]),
            "sig": bool(r["significant_at_0.05"]),
            "sig_holm": bool(r["significant_holm"]),
        })
    out["n_tests"] = len(out["rows"])
    out["n_sig"] = sum(1 for r in out["rows"] if r["sig"])
    out["n_sig_holm"] = sum(1 for r in out["rows"] if r["sig_holm"])
    # The comparisons that came back NOT significant are the informative ones:
    # everything else is "a big gap is a big gap". With ~10,000 paired rows even
    # a fraction of a percent reaches significance, so a null result here means
    # the two systems really are hard to tell apart.
    out["null_results"] = [r for r in out["rows"] if not r["sig"]]
    return out


def leakage_block():
    p = EXTRA / "leakage_report.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    corpus = df[df.check == "corpus_overlap_with_isot"][["test", "pct_of_test"]]
    dups = df[df.check == "within_corpus_duplicates"][["test", "pct_of_test"]]
    tt = df[(df.check == "train_test_overlap") &
            (~df.get("by_design_full_pool", False).astype(bool))]
    # How much of each test set a word-counter alone could solve. Stored under
    # pct_of_test by the leakage command, but it is an AUC, not a percentage.
    ln = df[df.check == "length_shortcut"][["test", "pct_of_test"]]
    return {
        "corpus": [{"name": r["test"], "pct": r["pct_of_test"]} for _, r in corpus.iterrows()],
        "dups": [{"name": r["test"], "pct": r["pct_of_test"]} for _, r in dups.iterrows()],
        "worst_tt": round(float(tt.pct_of_test.max()), 3) if len(tt) else 0.0,
        "length_auc": [{"name": r["test"], "auc": r["pct_of_test"]} for _, r in ln.iterrows()],
    }


def lengthcontrol_block():
    """real_syn against its length-controlled twin, on every test set available.

    The pair exists to answer one question: how much of what real_syn scores was
    ever about the FACTS, and how much was about the fake class being a
    different length from the real class at test time. Reporting it on LIAR
    alone would not settle that -- LIAR is separable at AUC 0.9999 by word count
    alone, so a drop there could equally mean "the model got worse". The WELFake
    columns are the control: length is uninformative there (AUC ~0.44), so if
    the length-controlled model holds up on WELFake while collapsing on LIAR,
    the LIAR drop is specifically the loss of the length cue and not a weaker
    model.
    """
    pair = ["real_syn", "real_syn_mixedlen"]
    if not _metrics_json("LR", "real_syn_mixedlen"):
        return {}
    out = {"models": MODELS, "recipes": pair, "tests": {}}
    # LIAR comes from the per-run metrics files train.py writes.
    out["tests"]["LIAR"] = {c: {m: _metrics_json(m, c) for m in MODELS} for c in pair}
    for label, fname in (("WELFake (as shipped)", "crossdomain2_results.csv"),
                         ("WELFake (ISOT removed)", "crosstarget_welfake_clean_results.csv")):
        p = EXTRA / fname
        if not p.exists():
            continue
        df = pd.read_csv(p)
        block = {}
        for c in pair:
            g = df[df.comp == c]
            if g.empty:
                continue
            block[c] = {r["model"]: {k: round(float(r[k]), 4) for k in METRICS if k in r}
                        for _, r in g.iterrows()}
        if len(block) == len(pair):     # only show the comparison if both sides ran
            out["tests"][label] = block
    return out


def demo_examples():
    """Two ready-made inputs for the try-it box, so it is usable without the
    visitor having to find an article first.

    Taken from the project's own data: a genuine held-out ISOT article and one
    of the synthetic fakes, which makes the demo show the actual thing this
    thesis is about rather than arbitrary text.
    """
    out = []
    p = cfg.PROCESSED_DIR / "test_crossdomain.csv"
    if p.exists():
        df = pd.read_csv(p)
        real = df[(df.label == 0) & (df.text.astype(str).str.len().between(900, 2200))]
        if len(real):
            out.append({"label": "Genuine article (held-out ISOT)",
                        "kind": "real",
                        "text": str(real.sample(1, random_state=cfg.SEED).iloc[0]["text"])})
    p = cfg.SYNTHETIC_DIR / "synthetic_fake.csv"
    if p.exists():
        df = pd.read_csv(p)
        fake = df[df.text.astype(str).str.len().between(900, 2200)]
        if len(fake):
            r = fake.sample(1, random_state=cfg.SEED).iloc[0]
            out.append({"label": "AI-generated fake (one fact altered)",
                        "kind": "fake",
                        "text": str(r["text"]),
                        "changed": str(r.get("modified_fact", ""))})
    return out


def collect():
    rq1_comps = ["real_real", "mixed", "real_syn",
                 "c2_synreal_realfake", "c3_synreal_synfake"]
    # The length-controlled twin of real_syn sits beside it rather than
    # replacing it: the two differ in exactly one variable, so the pair is what
    # carries the finding. Appended only if it has actually been trained, so a
    # checkout without that run still builds a complete report.
    if _metrics_json("LR", "real_syn_mixedlen"):
        rq1_comps.append("real_syn_mixedlen")
    rq3_comps = rq1_comps + ["real_syn_multisource"]
    return {
        "models": MODELS,
        "labels": RECIPE_LABEL,
        "metrics": METRICS,
        # Both test sets, for the same reason RQ2 now shows both: a score on
        # LIAR alone can't distinguish "detected the fake" from "noticed it was
        # short". Cleaned WELFake covers every recipe here and is length-neutral.
        "rq1": {"comps": rq1_comps, "liar": liar_block(rq1_comps),
                "welfake": welfake_block(rq1_comps),
                "tests": {"LIAR": liar_block(rq1_comps),
                          "WELFake (ISOT removed)": welfake_clean_block(rq1_comps)},
                "cv": cv_block(rq1_comps + ["style_robust"]),
                "significance": significance_block(),
                "editsym": editsym_block()},
        "rq2": sweep_block(),
        # RQ3 is the model-family comparison; the LIAR-vs-WELFake material it
        # used to hold is the cross-domain protocol common to all four
        # questions, so it lives under "framework" instead of being one RQ.
        "rq3": {"families": families_block(rq3_comps), "seeds": seed_block(),
                "seedRuns": seed_runs_block()},
        "rq4": style_block(),
        "framework": {"comps": rq3_comps, "liar": liar_block(rq3_comps),
                      "welfake": welfake_block(rq3_comps), "length": length_block(),
                      "quality": quality_block(), "leakage": leakage_block(),
                      "contamination": contamination_block(rq1_comps),
                      "lengthcontrol": lengthcontrol_block()},
        "demo": demo_examples(),
    }


def main():
    data = collect()
    tpl = (cfg.ROOT / "src" / "report_template.html").read_text(encoding="utf-8")
    html = tpl.replace("/*__DATA__*/null", json.dumps(data, indent=None))

    # Inline the detector rather than linking it. The page has to work when
    # opened straight off disk, where fetch() is blocked by CORS and an
    # external <script src> is not reliably executed -- and the report's whole
    # point is being one file you can send someone. Costs ~1.5 MB.
    model = cfg.ROOT / "detector_model.js"
    scorer = cfg.ROOT / "src" / "detector.js"
    if model.exists() and scorer.exists():
        blob = model.read_text(encoding="utf-8") + "\n" + scorer.read_text(encoding="utf-8")
        html = html.replace("/*__DETECTOR__*/", blob)
        det = f"inlined ({len(blob)/1024/1024:.2f} MB)"
    else:
        # Leave the placeholder empty; the tab detects the missing model and
        # says so instead of throwing.
        html = html.replace("/*__DETECTOR__*/", "")
        det = "MISSING -- run src/export_detector_model.py first"

    out = cfg.ROOT / "results_report.html"
    out.write_text(html, encoding="utf-8")
    kb = len(html) / 1024
    print(f"Wrote {out} ({kb:.0f} KB)")
    print(f"  detector       : {det}")
    print(f"  demo examples  : {len(data.get('demo', []))}")
    print(f"  RQ1 recipes    : {len(data['rq1']['liar'])}")
    print(f"  RQ3 families   : {len(data['rq3']['families']['byModel'])}")
    print(f"  seed rows      : {len(data['rq3']['seeds'])}")
    print(f"  sweep points   : {len(data['rq2']['fractions'])}")
    print(f"  framework rows : {len(data['framework']['liar'])}")


if __name__ == "__main__":
    main()
