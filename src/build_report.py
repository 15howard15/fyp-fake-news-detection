
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
                "seedRuns": seed_runs_block(), "matched": matched_block()},
        "rq4": style_block(),
        "framework": {"comps": rq3_comps, "liar": liar_block(rq3_comps),
                      "welfake": welfake_block(rq3_comps), "length": length_block(),
                      "quality": quality_block(), "leakage": leakage_block()},
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
