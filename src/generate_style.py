"""
generate_style.py -- all three LLM tone-rewriting passes for Objective 4.

Merged from generate_style_attack.py, generate_style_attack_reverse.py and
generate_counter_style_training.py, which were three near-identical scripts:
the same generate_one(), the same resume-from-CSV logic, and the same
two-loop real-pool/fake-pool structure, differing only in which prompt each
class gets, where the source articles come from, and where the output goes.

Three subcommands:

  attack           TEST-side, forward direction. Real -> sensationalized,
                   fake -> neutralized. This is the direction that matches the
                   intuitive "fake news sounds dramatic" shortcut, so it is
                   also the easiest direction to defend against by design --
                   which is why `attack-reverse` exists.

  attack-reverse   TEST-side, opposite direction. Real -> neutralized,
                   fake -> sensationalized. Checks that the style_robust fix
                   generalises to a tone shift it was not built against rather
                   than memorising one attack pattern. Draws from the same
                   held-out pools but explicitly excludes every article the
                   forward attack already used, read back from
                   style_attack_originals.csv rather than assumed from an index
                   offset, since quality-filter retries mean the forward run's
                   stopping index is not fixed.

  counter-training TRAINING-side fix. Paired counter-style twins of articles
                   ALREADY in train_real_real.csv, each keeping its original
                   label, so the model sees the same content in both tones
                   under the same label. Decorrelates tone from truth during
                   training rather than testing for the correlation.

All three reuse gen_common's STYLE_TRANSFER_SYSTEM_PROMPT, so training-time and
test-time style shifts are drawn from the same distribution.

Usage:
    python src/generate_style.py attack --n_per_class 100
    python src/generate_style.py attack-reverse --n_per_class 100
    python src/generate_style.py counter-training --n_per_class 100
"""
import argparse
import os

import pandas as pd
from tqdm import tqdm

import config as cfg
from gen_common import (
    truncate_article, quality_ok, call_llm, STYLE_TRANSFER_SYSTEM_PROMPT,
    SENSATIONALIZE_REAL_TEMPLATE, NEUTRALIZE_FAKE_TEMPLATE,
)

NEUTRALIZE_REAL_TEMPLATE = """Source article (this is REAL, factual news):
\"\"\"{article}\"\"\"

Rewrite this article using calm, neutral, measured, factual-sounding
journalistic tone -- plain wire-service style, no embellishment. Do NOT
change, add, or remove any fact, name, date, or number. The rewritten
article must remain exactly as true as the original.

Return JSON with exactly these keys:
{{
  "styled_article": "the full rewritten article text"
}}"""

SENSATIONALIZE_FAKE_TEMPLATE = """Source article (this is FAKE / false news):
\"\"\"{article}\"\"\"

Rewrite this article using sensational, alarmist, emotionally charged
language -- as if a tabloid or clickbait outlet were reporting the SAME
false story. Exaggerate tone and framing only. Do NOT change, add, or
remove any fact, name, date, or number, and do NOT correct the false
claim -- the rewritten article must remain exactly as FALSE as the
original, just more dramatic in tone.

Return JSON with exactly these keys:
{{
  "styled_article": "the full rewritten article text"
}}"""


def generate_one(client, article: str, template: str):
    prompt = template.format(article=truncate_article(article))
    result = call_llm(client, STYLE_TRANSFER_SYSTEM_PROMPT, prompt, "styled_article")
    return result["styled_article"] if result else None


def _open_client():
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("Set OPENAI_API_KEY first (see README).")
    from openai import OpenAI
    return OpenAI()


def _run_pass(client, pool, template, tag, label, n, results, pbar, out_path,
              *, start_idx, track_orig_id, kind, skip_idx=frozenset(),
              done_ids=frozenset()):
    """One class's rewriting loop, shared by all three subcommands.

    start_idx is the caller's resume position, and it is deliberately NOT
    unified across subcommands: `attack` and `counter-training` resume by
    offset (start at the count already done), while `attack-reverse` restarts
    at 0 and skips by ID instead, because it must also skip whatever the
    forward attack consumed. Collapsing these into one scheme would silently
    re-generate or skip articles -- both of which cost API credit and change
    the corpus a published result was computed from.
    """
    idx = start_idx
    made = len([r for r in results if r["attack_type"] == tag])
    while made < n and idx < len(pool):
        orig_id = f"{kind}_{idx}"
        if idx in skip_idx or orig_id in done_ids:
            idx += 1
            continue
        article = str(pool.iloc[idx]["text"])
        idx += 1
        if len(article.split()) < 30:
            continue
        styled = generate_one(client, article, template)
        if styled is None or not quality_ok(article, styled, ratio_range=(0.4, 2.5)):
            continue
        if track_orig_id:
            row = {"orig_id": orig_id, "text": styled, "label": label,
                   "source_text": truncate_article(article, cfg.FULL_SOURCE_CAP),
                   "attack_type": tag}
        else:
            row = {"text": styled, "label": label,
                   "source": "counter_style", "attack_type": tag}
        results.append(row)
        made += 1
        pbar.update(1)
        if len(results) % 20 == 0:
            pd.DataFrame(results).to_csv(out_path, index=False)


def _save_originals(results, real_pool, fake_pool, path):
    """The matching unattacked articles, so eval can pair before/after."""
    rows = []
    for r in results:
        kind, i = r["orig_id"].split("_")
        pool = real_pool if kind == "real" else fake_pool
        i = int(i)
        rows.append({"orig_id": r["orig_id"], "text": pool.iloc[i]["text"],
                     "label": pool.iloc[i]["label"], "attack_type": r["attack_type"]})
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved matching originals to {path}")


def _resume(out_path, real_tag, fake_tag):
    results, done_real, done_fake = [], 0, 0
    if out_path.exists():
        results = pd.read_csv(out_path).to_dict("records")
        done_real = sum(1 for r in results if r["attack_type"] == real_tag)
        done_fake = sum(1 for r in results if r["attack_type"] == fake_tag)
        print(f"Resuming: {done_real} real / {done_fake} fake already done.")
    return results, done_real, done_fake


def _finish(results, out_path, noun):
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} {noun} to {out_path}")
    print(df["attack_type"].value_counts().to_string())
    return df


def cmd_attack(args, reverse=False):
    """TEST-side attack set, drawn from held-out test_indomain."""
    client = _open_client()
    test_df = pd.read_csv(cfg.PROCESSED_DIR / "test_indomain.csv")
    real_pool = test_df[test_df.label == cfg.LABEL_REAL].sample(
        frac=1.0, random_state=cfg.SEED).reset_index(drop=True)
    fake_pool = test_df[test_df.label == cfg.LABEL_FAKE].sample(
        frac=1.0, random_state=cfg.SEED).reset_index(drop=True)

    if reverse:
        real_tpl, fake_tpl = NEUTRALIZE_REAL_TEMPLATE, SENSATIONALIZE_FAKE_TEMPLATE
        real_tag, fake_tag = "neutralize_real", "sensationalize_fake"
        out_path = cfg.SYNTHETIC_DIR / "style_attack_reverse.csv"
        orig_path = cfg.SYNTHETIC_DIR / "style_attack_reverse_originals.csv"
        desc, noun = "Reverse style attacks", "reverse style-attacked articles"
    else:
        real_tpl, fake_tpl = SENSATIONALIZE_REAL_TEMPLATE, NEUTRALIZE_FAKE_TEMPLATE
        real_tag, fake_tag = "sensationalize", "neutralize"
        out_path = cfg.SYNTHETIC_DIR / "style_attack.csv"
        orig_path = cfg.SYNTHETIC_DIR / "style_attack_originals.csv"
        desc, noun = "Style attacks", "style-attacked articles"

    used_real, used_fake = set(), set()
    if reverse:
        fwd = cfg.SYNTHETIC_DIR / "style_attack_originals.csv"
        if fwd.exists():
            for oid in pd.read_csv(fwd)["orig_id"]:
                kind, i = oid.split("_")
                (used_real if kind == "real" else used_fake).add(int(i))
            print(f"Excluding {len(used_real)} real / {len(used_fake)} fake "
                  f"articles already used by the forward-direction attack.")

    results, done_real, done_fake = _resume(out_path, real_tag, fake_tag)
    done_ids = {r["orig_id"] for r in results} if reverse else frozenset()
    pbar = tqdm(total=args.n_per_class * 2,
                initial=done_real + done_fake, desc=desc)

    _run_pass(client, real_pool, real_tpl, real_tag, cfg.LABEL_REAL,
              args.n_per_class, results, pbar, out_path,
              start_idx=0 if reverse else done_real, track_orig_id=True,
              kind="real", skip_idx=used_real, done_ids=done_ids)
    _run_pass(client, fake_pool, fake_tpl, fake_tag, cfg.LABEL_FAKE,
              args.n_per_class, results, pbar, out_path,
              start_idx=0 if reverse else done_fake, track_orig_id=True,
              kind="fake", skip_idx=used_fake, done_ids=done_ids)

    pbar.close()
    _finish(results, out_path, noun)
    _save_originals(results, real_pool, fake_pool, orig_path)


def cmd_counter_training(args):
    """TRAINING-side fix: counter-style twins of train_real_real rows.

    Source pool is the training set, NOT the held-out test pool the attack
    sets use, so the fix cannot be credited to having seen the test articles.
    """
    client = _open_client()
    train = pd.read_csv(cfg.PROCESSED_DIR / "train_real_real.csv")
    real_pool = train[train.label == cfg.LABEL_REAL].reset_index(drop=True)
    fake_pool = train[train.label == cfg.LABEL_FAKE].reset_index(drop=True)
    print(f"train_real_real: {len(real_pool)} real / {len(fake_pool)} fake "
          f"available to pair from")

    out_path = cfg.SYNTHETIC_DIR / "counter_style_training.csv"
    results, done_real, done_fake = _resume(out_path, "sensationalize", "neutralize")
    pbar = tqdm(total=args.n_per_class * 2, initial=done_real + done_fake,
                desc="Counter-style training pairs")

    _run_pass(client, real_pool, SENSATIONALIZE_REAL_TEMPLATE, "sensationalize",
              cfg.LABEL_REAL, args.n_per_class, results, pbar, out_path,
              start_idx=done_real, track_orig_id=False, kind="real")
    _run_pass(client, fake_pool, NEUTRALIZE_FAKE_TEMPLATE, "neutralize",
              cfg.LABEL_FAKE, args.n_per_class, results, pbar, out_path,
              start_idx=done_fake, track_orig_id=False, kind="fake")

    pbar.close()
    _finish(results, out_path, "counter-style training pairs")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="command", required=True)
    for name, helptext in [
        ("attack", "TEST-side forward attack: real->sensational, fake->neutral"),
        ("attack-reverse", "TEST-side reverse attack: real->neutral, fake->sensational"),
        ("counter-training", "TRAINING-side fix: counter-style twins of train_real_real"),
    ]:
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--n_per_class", type=int, default=100,
                       help="number of real and number of fake articles (each)")

    args = ap.parse_args()
    if args.command == "attack":
        cmd_attack(args, reverse=False)
    elif args.command == "attack-reverse":
        cmd_attack(args, reverse=True)
    else:
        cmd_counter_training(args)


if __name__ == "__main__":
    main()
