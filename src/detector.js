
(function (global) {
  "use strict";

  let M = null, STOP = null;

  function ready() {
    if (M) return true;
    const raw = global.DETECTOR_MODEL;
    if (!raw) return false;
    M = { idf: raw.idf, coef: raw.coef, intercept: raw.intercept, meta: raw.meta };

    M.vocab = new Map();
    for (let i = 0; i < raw.terms.length; i++) M.vocab.set(raw.terms[i], i);
    STOP = new Set(raw.stopwords);
    return true;
  }

    function clean(text) {
    let t = String(text);
    const ta = document.createElement("textarea");
    ta.innerHTML = t; t = ta.value;
    t = t.replace(/http\S+|www\.\S+/g, " ");
    t = t.replace(/<[^>]+>/g, " ");
    t = t.toLowerCase();
    t = t.replace(/[^a-z0-9\s]/g, " ");
    t = t.replace(/\s+/g, " ").trim();
    return t.split(" ").filter(w => w && !STOP.has(w)).join(" ");
  }

    function tokens(cleaned) {
    return cleaned.split(" ").filter(w => w.length >= 2);
  }

  function counts(toks) {
    const c = new Map();
    const bump = t => { const i = M.vocab.get(t); if (i !== undefined) c.set(i, (c.get(i) || 0) + 1); };
    for (let i = 0; i < toks.length; i++) bump(toks[i]);
    for (let i = 0; i + 1 < toks.length; i++) bump(toks[i] + " " + toks[i + 1]);
    return c;
  }

    function score(text) {
    if (!ready()) return null;
    const cleaned = clean(text);
    const toks = tokens(cleaned);
    const c = counts(toks);
    if (c.size === 0) return { empty: true, tokens: toks.length };

    let norm = 0;
    const weighted = new Map();
    c.forEach((tf, i) => { const v = tf * M.idf[i]; weighted.set(i, v); norm += v * v; });
    norm = Math.sqrt(norm) || 1;

    let z = M.intercept;
    const contrib = [];
    weighted.forEach((v, i) => {
      const x = v / norm, w = x * M.coef[i];
      z += w;
      contrib.push([i, w]);
    });

    contrib.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
    const terms = global.DETECTOR_MODEL.terms;
    const top = contrib.slice(0, 8).map(([i, w]) => ({ term: terms[i], weight: w }));

    return {
      prob_fake: 1 / (1 + Math.exp(-z)),
      z: z,
      tokens: toks.length,
      matched: c.size,
      top: top,
      meta: M.meta,
    };
  }

  global.Detector = { score, ready, clean };
})(window);
