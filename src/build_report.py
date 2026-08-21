"""build_report.py -- regenerate results_report.html from the results files."""
import json

import pandas as pd
from sklearn.metrics import roc_auc_score

import config as cfg

RESULTS = cfg.RESULTS_DIR
EXTRA = RESULTS / "extra"
MODELS = ["LR", "SVM", "CNN", "BERT"]

RECIPE_LABEL = {
    "real_real": "Real + Real",
    "half_synthetic": "Real + Mixed",
    "full_synthetic": "Real + Synthetic",
    "synthetic_real_only": "Synthetic-real + Real-fake",
    "both_synthetic": "Synthetic-real + Synthetic-fake",
    "synthetic_multisource": "Real + Synthetic (diverse-sourced)",
    "synthetic_length_controlled": "Real + Synthetic (length-controlled)",
    "synthetic_real_only_matched": "Synthetic-real + Real-fake (edit-matched)",
    "both_synthetic_matched": "Synthetic-real + Synthetic-fake (edit-matched)",
    "style_robust": "Style-robust",
}
METRICS = ["f1", "auc_roc", "accuracy", "precision", "recall"]

RECIPE_PLAIN = {
    "real_real": "Real news + real fake news",
    "half_synthetic": "Real news + half-synthetic fakes",
    "full_synthetic": "Real news + all-synthetic fakes",
    "synthetic_real_only": "AI-rewritten real news + real fake news",
    "both_synthetic": "AI-rewritten real news + AI-written fakes",
    "synthetic_multisource": "Real news + synthetic fakes from mixed sources",
    "synthetic_length_controlled": "Real news + length-matched synthetic fakes",
    "style_robust": "Tone-balanced training (the fix)",
    "synthetic_real_only_matched": "AI-rewritten real + real fake, edit-matched",
    "both_synthetic_matched": "Both classes AI-written, edit-matched",
    "synthetic_25pct": "Real news + 25% synthetic fakes",
    "synthetic_75pct": "Real news + 75% synthetic fakes",
}

RECIPE_ROLE = {
    "real_real": "control",
    "half_synthetic": "control",
    "full_synthetic": "failing",
    "synthetic_real_only": "control",
    "both_synthetic": "failing",
    "synthetic_multisource": "failing",
    "synthetic_length_controlled": "optimized",
    "style_robust": "optimized",
    "synthetic_real_only_matched": "optimized",
    "both_synthetic_matched": "optimized",
}


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


SWEEP_PCT = {"synthetic_0pct": "0%", "synthetic_25pct": "25%", "synthetic_50pct": "50%",
             "synthetic_75pct": "75%", "synthetic_100pct": "100%"}
SWEEP_ALIAS = {"synthetic_0pct": "real_real", "synthetic_50pct": "half_synthetic", "synthetic_100pct": "full_synthetic"}

SWEEP_METRICS = ["f1", "auc_roc", "precision", "recall"]


def sweep_block():
    """RQ2 synthetic-fraction sweep, on every test set it has been scored against."""
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

    files = {"synthetic_0pct": "train_real_real", "synthetic_25pct": "train_synthetic_25pct",
             "synthetic_50pct": "train_half_synthetic", "synthetic_75pct": "train_synthetic_75pct",
             "synthetic_100pct": "train_full_synthetic"}
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

    cv = cv_block([SWEEP_ALIAS.get(s, s) for s in SWEEP_PCT])
    if cv.get("comps"):
        by_frac = {}
        for sweep, pct in SWEEP_PCT.items():
            comp = SWEEP_ALIAS.get(sweep, sweep)
            if comp in cv["rows"]:
                by_frac[pct] = cv["rows"][comp]
        out["cv"] = {"fractions": [p for p in SWEEP_PCT.values() if p in by_frac],
                     "rows": by_frac, "splits": cv["splits"]}

    summary = {}
    for label, blk in out["tests"].items():
        rows = []
        for i, frac in enumerate(order):
            deltas = []
            for m in MODELS:
                vals = blk.get(m, {}).get("f1") or []
                if i < len(vals) and vals[i] is not None and vals[0] is not None:
                    deltas.append(vals[i] - vals[0])
            if deltas:
                rows.append({"fraction": frac,
                             "mean_delta": round(sum(deltas) / len(deltas), 4),
                             "worst_delta": round(min(deltas), 4),
                             "n_models": len(deltas)})
        summary[label] = rows
    out["summary"] = summary

    SAFE_DROP = 0.15
    cliffs = {}
    for label, rows in summary.items():
        cliff = None
        for r in rows:
            if r["worst_delta"] < -SAFE_DROP:
                cliff = r["fraction"]
                break
        cliffs[label] = cliff
    out["cliff"] = cliffs
    out["safe_drop"] = SAFE_DROP

    out["series"] = out["tests"].get("LIAR", {})
    return out


def style_block():
    cur = pd.read_csv(EXTRA / "style_robustness_results.csv")
    rev = pd.read_csv(EXTRA / "style_robustness_reverse_results.csv")
    recipes = ["real_real", "half_synthetic", "full_synthetic", "style_robust"]
    flips = {RECIPE_LABEL[c]: {m: None for m in MODELS} for c in recipes}
    for _, r in cur.iterrows():
        if r["comp"] in recipes:
            flips[RECIPE_LABEL[r["comp"]]][r["model"]] = round(float(r["flip_rate"]), 4)
    reverse = {}
    for _, r in rev[rev.comp == "style_robust"].iterrows():
        reverse[r["model"]] = round(float(r["flip_rate"]), 4)
    original = {m: flips["Style-robust"][m] for m in MODELS}

    DEGENERATE_AT = 0.95
    degenerate = {}
    for comp in ("real_real", "half_synthetic", "full_synthetic", "style_robust"):
        label = RECIPE_LABEL.get(comp, comp)
        row = {}
        for m in MODELS:
            p = cfg.RESULTS_DIR / f"metrics_{m}_{comp}.json"
            if not p.exists():
                continue
            c = json.loads(p.read_text()).get("confusion")
            if not c:
                continue
            n = c["tn"] + c["fp"] + c["fn"] + c["tp"]
            pred_fake = c["tp"] + c["fp"]
            share = max(pred_fake, n - pred_fake) / max(n, 1)
            row[m] = {"one_class_share": round(share, 4),
                      "degenerate": bool(share >= DEGENERATE_AT)}
        if row:
            degenerate[label] = row
    baseline = {}
    for comp in ("real_real", "style_robust"):
        label = RECIPE_LABEL[comp]
        row = {}
        for m in MODELS:
            r = cur[(cur.comp == comp) & (cur.model == m)]
            if r.empty:
                continue
            r = r.iloc[0]
            row[m] = {
                "accuracy": round(float(r["acc_original"]), 4),
                "precision": round(float(r["precision_original"]), 4),
                "recall": round(float(r["recall_original"]), 4),
                "f1": round(float(r["f1_original"]), 4),
                "auc_roc": round(float(r["auc_original"]), 4),
            }
        baseline[label] = row
    return {"flips": flips, "original": original, "reverse": reverse,
            "baseline": baseline, "degenerate": degenerate,
            "degenerate_at": DEGENERATE_AT}


def families_block(comps):
    """RQ3 -- model-family consistency, which has two independent halves."""
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
    """The individual per-seed F1 scores, not just their summary."""
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


def editsym_block():
    pairs = [("synthetic_real_only", "synthetic_real_only_matched"), ("both_synthetic", "both_synthetic_matched")]
    if not _metrics_json("LR", "both_synthetic_matched"):
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

    p = EXTRA / "edit_distance.csv"
    if p.exists():
        df = pd.read_csv(p).set_index("file")["similarity_mean"].to_dict()
        need = ("synthetic_real_sym", "synthetic_real", "synthetic_fake",
                "synthetic_fake_sym")
        if all(k in df and pd.notna(df[k]) for k in need):
            r_before, r_after = float(df["synthetic_real"]), float(df["synthetic_real_sym"])
            f_before, f_after = float(df["synthetic_fake"]), float(df["synthetic_fake_sym"])
            out["edit"] = {
                "real": round(r_after, 4),
                "real_before": round(r_before, 4),
                "fake_before": round(f_before, 4),
                "fake_after": round(f_after, 4),
                "gap_before": round(abs(r_before - f_before) * 100, 1),
                "gap_after": round(abs(r_after - f_after) * 100, 1)}
    return out


def cv_block(comps):
    """5-fold CV for LR/SVM, under both an ordinary and a pair-aware split."""
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
    out["null_results"] = [r for r in out["rows"] if not r["sig"]]
    return out


def demo_examples():
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
    rq1_comps = ["real_real", "half_synthetic", "full_synthetic",
                 "synthetic_real_only", "both_synthetic"]
    if _metrics_json("LR", "synthetic_length_controlled"):
        rq1_comps.append("synthetic_length_controlled")
    rq3_comps = rq1_comps + ["synthetic_multisource"]
    return {
        "models": MODELS,
        "labels": RECIPE_LABEL,
        "roles": RECIPE_ROLE,
        "plain": dict(RECIPE_PLAIN,
                      **{RECIPE_LABEL[c]: p for c, p in RECIPE_PLAIN.items()
                         if c in RECIPE_LABEL}),
        "metrics": METRICS,
        "rq1": {"comps": rq1_comps, "liar": liar_block(rq1_comps),
                "welfake": welfake_block(rq1_comps),
                "tests": {"LIAR": liar_block(rq1_comps),
                          "WELFake (ISOT removed)": welfake_clean_block(rq1_comps)},
                "significance": significance_block(),
                "editsym": editsym_block()},
        "rq2": sweep_block(),
        "rq3": {"families": families_block(rq3_comps), "seeds": seed_block(),
                "seedRuns": seed_runs_block()},
        "rq4": style_block(),
        "demo": demo_examples(),
    }


def main():
    data = collect()
    tpl = (cfg.ROOT / "src" / "report_template.html").read_text(encoding="utf-8")
    html = tpl.replace("/*__DATA__*/null", json.dumps(data, indent=None))

    model = cfg.ROOT / "detector_model.js"
    scorer = cfg.ROOT / "src" / "detector.js"
    if model.exists() and scorer.exists():
        blob = model.read_text(encoding="utf-8") + "\n" + scorer.read_text(encoding="utf-8")
        html = html.replace("/*__DETECTOR__*/", blob)
        det = f"inlined ({len(blob)/1024/1024:.2f} MB)"
    else:
        html = html.replace("/*__DETECTOR__*/", "")
        det = "MISSING -- run src/detector.py export first"

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


if __name__ == "__main__":
    main()
