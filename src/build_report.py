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

Layout: one tab per research question, plus a validity tab for the checks that
cut across all four (seed stability, matched-length comparison, data quality,
train/test leakage). Each RQ tab states the question, gives the short answer,
then shows the evidence with a metric selector so accuracy / precision /
recall / F1 / AUC-ROC are all reachable rather than F1 alone.

Colours are the dataviz reference palette used unmodified in slot order
(blue / orange / aqua / yellow for the four models). Slots 3 and 4 sit below
3:1 contrast on the light surface, so the relief rule applies: every bar
carries a direct value label and every chart has a full data table beneath it.
"""
import json

import pandas as pd

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


def welfake_block(comps):
    p = EXTRA / "crossdomain2_results.csv"
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


def sweep_block():
    """RQ2 synthetic-fraction sweep, cross-domain only."""
    df = pd.read_csv(EXTRA / "swap_sweep_results.csv")
    df = df[df.test == "crossdomain"]
    pct = {"swap_000": "0%", "swap_025": "25%", "swap_050": "50%",
           "swap_075": "75%", "swap_100": "100%"}
    order = list(pct.values())
    out = {"fractions": order, "series": {}}
    for m in MODELS:
        g = df[df.model == m]
        by = {pct[r["sweep"]]: r for _, r in g.iterrows() if r["sweep"] in pct}
        out["series"][m] = {
            met: [round(float(by[f][met]), 4) if f in by and met in by[f] else None
                  for f in order]
            for met in ["f1", "auc_roc", "precision", "recall"]
        }
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


def matched_block():
    p = EXTRA / "matched_length_results.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    out = {}
    for comp, g in df.groupby("comp"):
        out[RECIPE_LABEL.get(comp, comp)] = {
            r["model"]: {"f1_full": round(float(r["f1_full"]), 4),
                         "f1_max300": round(float(r["f1_max300"]), 4),
                         "auc_full": round(float(r["auc_full"]), 4),
                         "auc_max300": round(float(r["auc_max300"]), 4)}
            for _, r in g.iterrows()}
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


def leakage_block():
    p = EXTRA / "leakage_report.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    corpus = df[df.check == "corpus_overlap_with_isot"][["test", "pct_of_test"]]
    dups = df[df.check == "within_corpus_duplicates"][["test", "pct_of_test"]]
    tt = df[(df.check == "train_test_overlap") &
            (~df.get("by_design_full_pool", False).astype(bool))]
    return {
        "corpus": [{"name": r["test"], "pct": r["pct_of_test"]} for _, r in corpus.iterrows()],
        "dups": [{"name": r["test"], "pct": r["pct_of_test"]} for _, r in dups.iterrows()],
        "worst_tt": round(float(tt.pct_of_test.max()), 3) if len(tt) else 0.0,
    }


def collect():
    rq1_comps = ["real_real", "mixed", "real_syn",
                 "c2_synreal_realfake", "c3_synreal_synfake"]
    rq3_comps = rq1_comps + ["real_syn_multisource"]
    return {
        "models": MODELS,
        "labels": RECIPE_LABEL,
        "metrics": METRICS,
        "rq1": {"comps": rq1_comps, "liar": liar_block(rq1_comps),
                "welfake": welfake_block(rq1_comps)},
        "rq2": sweep_block(),
        # RQ3 is the model-family comparison; the LIAR-vs-WELFake material it
        # used to hold is the cross-domain protocol common to all four
        # questions, so it lives under "framework" instead of being one RQ.
        "rq3": {"families": families_block(rq3_comps), "seeds": seed_block(),
                "matched": matched_block()},
        "rq4": style_block(),
        "framework": {"comps": rq3_comps, "liar": liar_block(rq3_comps),
                      "welfake": welfake_block(rq3_comps), "length": length_block(),
                      "quality": quality_block(), "leakage": leakage_block()},
    }


def main():
    data = collect()
    tpl = (cfg.ROOT / "src" / "report_template.html").read_text(encoding="utf-8")
    html = tpl.replace("/*__DATA__*/null", json.dumps(data, indent=None))
    out = cfg.ROOT / "results_report.html"
    out.write_text(html, encoding="utf-8")
    kb = len(html) / 1024
    print(f"Wrote {out} ({kb:.0f} KB)")
    print(f"  RQ1 recipes    : {len(data['rq1']['liar'])}")
    print(f"  RQ3 families   : {len(data['rq3']['families']['byModel'])}")
    print(f"  seed rows      : {len(data['rq3']['seeds'])}")
    print(f"  sweep points   : {len(data['rq2']['fractions'])}")
    print(f"  framework rows : {len(data['framework']['liar'])}")


if __name__ == "__main__":
    main()
