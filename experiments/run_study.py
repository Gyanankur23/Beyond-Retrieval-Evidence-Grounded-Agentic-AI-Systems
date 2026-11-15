"""Version-aware retrieval study on real versioned documentation (no LLMs, no external APIs).

Exp A  Does a version-blind retriever return the wrong version of a section? (outdated-document failure)
Exp B  How often is that wrong version *materially different* ("stale hit")? (conflicting-source risk)
Exp C  Can the top retrieval score be used to abstain on out-of-corpus queries? (insufficient evidence)

Everything is deterministic (seed 0). Outputs: results/*.csv, results/figures/*.png, results/macros.tex
"""
import json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"; FIG = RES / "figures"; RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
SEED = 0
rng = np.random.default_rng(SEED)
secs = [json.loads(l) for l in open(ROOT / "data" / "sections.jsonl", encoding="utf-8")]
pairs = json.load(open(ROOT / "data" / "pairs.json"))
S = {s["id"]: s for s in secs}
LIBS = ["fastapi", "pandas", "sklearn", "pydantic"]
LIBNAME = {"fastapi": "FastAPI", "pandas": "pandas", "sklearn": "scikit-learn", "pydantic": "Pydantic"}


# ------------------------------------------------------------------ retrievers
class Retriever:
    def __init__(self, kind, texts):
        self.kind = kind
        if kind == "bm25":
            self.cv = CountVectorizer(stop_words="english"); X = self.cv.fit_transform(texts).tocsr().astype(float)
            N = X.shape[0]; df = np.asarray((X > 0).sum(0)).ravel(); idf = np.log((N - df + 0.5) / (df + 0.5) + 1)
            dl = np.asarray(X.sum(1)).ravel(); k1, b = 1.5, 0.75
            X = X.tocoo(); denom = X.data + k1 * (1 - b + b * dl[X.row] / dl.mean())
            from scipy.sparse import csr_matrix
            self.W = csr_matrix((X.data * (k1 + 1) / denom * idf[X.col], (X.row, X.col)), shape=X.shape)
        else:
            self.tv = TfidfVectorizer(stop_words="english", sublinear_tf=True); D = self.tv.fit_transform(texts)
            if kind == "tfidf":
                self.D = D
            else:  # lsa
                self.svd = TruncatedSVD(min(100, D.shape[0] - 1), random_state=SEED)
                self.Dl = normalize(self.svd.fit_transform(D))

    def scores(self, queries):
        if self.kind == "bm25":
            Q = self.cv.transform(queries); Q.data[:] = 1
            return (Q @ self.W.T).toarray()
        Q = self.tv.transform(queries)
        if self.kind == "tfidf":
            return (Q @ self.D.T).toarray()
        return normalize(self.svd.transform(Q)) @ self.Dl.T


def first_sentence(text, n=30):
    m = re.match(r"(.+?[.!?])(\s|$)", text)
    s = m.group(1) if m else text
    return " ".join(s.split()[:n])


def jac(a, b):
    a, b = set(re.findall(r"[a-z0-9]{2,}", a.lower())), set(re.findall(r"[a-z0-9]{2,}", b.lower()))
    return len(a & b) / max(len(a | b), 1)


def boot(x, B=5000):
    x = np.asarray(x, float)
    if len(x) < 2: return (np.nan, np.nan)
    m = np.random.default_rng(1).choice(x, (B, len(x))).mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


# ------------------------------------------------------------------ Exp A / B
rows = []
for lib in LIBS:
    ls = [s for s in secs if s["lib"] == lib]
    ids = np.array([s["id"] for s in ls]); pos = {i: k for k, i in enumerate(ids)}
    role = np.array([s["role"] for s in ls])
    texts = [s["heading"] + " " + s["text"] for s in ls]
    lp = [p for p in pairs if p["lib"] == lib and p["cls"] in ("changed", "unchanged")]
    for kind in ("bm25", "tfidf", "lsa"):
        R = Retriever(kind, texts)
        for qtype in ("heading", "heading+sentence"):
            qs, meta = [], []
            for p in lp:
                for tr, orole in (("old", "new"), ("new", "old")):
                    t = S[p[tr + "_id"]]
                    qs.append(t["heading"] if qtype == "heading" else t["heading"] + " " + first_sentence(t["text"]))
                    meta.append((p, tr, p[tr + "_id"], p[orole + "_id"]))
            sc = R.scores(qs) + rng.random((len(qs), len(ls))) * 1e-9          # random tie-break, fixed seed
            for mode in ("blind", "aware"):
                for i, (p, tr, tid, oid) in enumerate(meta):
                    s = sc[i].copy()
                    if mode == "aware":
                        s[role != tr] = -np.inf
                    top = int(np.argmax(s)); ti = pos[tid]
                    rank = 1 + int((s > s[ti]).sum())
                    top_id = ids[top]
                    rows.append(dict(lib=lib, retriever=kind, qtype=qtype, mode=mode, cls=p["cls"], jaccard=p["jaccard"],
                                     hit1=int(top_id == tid),
                                     equiv1=int(top_id == tid or jac(S[top_id]["text"], S[tid]["text"]) >= 0.95),
                                     stale1=int(top_id == oid and p["cls"] == "changed"),
                                     wrongver1=int(S[top_id]["role"] != tr),
                                     rr=1.0 / rank, rec5=int(rank <= 5)))
A = pd.DataFrame(rows); A.to_csv(RES / "exp_ab_per_query.csv.gz", index=False)

summ = []
for (cls, qt, rk, md), g in A.groupby(["cls", "qtype", "retriever", "mode"]):
    r = dict(cls=cls, qtype=qt, retriever=rk, mode=md, n=len(g))
    for m in ("hit1", "equiv1", "stale1", "wrongver1", "rec5", "rr"):
        lo, hi = boot(g[m]); r[m] = g[m].mean(); r[m + "_lo"] = lo; r[m + "_hi"] = hi
    summ.append(r)
SM = pd.DataFrame(summ); SM.to_csv(RES / "exp_ab_summary.csv", index=False)

tests = []
for (cls, qt, rk), g in A.groupby(["cls", "qtype", "retriever"]):
    b = g[g["mode"] == "blind"].reset_index(drop=True); a = g[g["mode"] == "aware"].reset_index(drop=True)
    n10 = int(((a.hit1 == 1) & (b.hit1 == 0)).sum()); n01 = int(((a.hit1 == 0) & (b.hit1 == 1)).sum())
    p = stats.binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    tests.append(dict(cls=cls, qtype=qt, retriever=rk, aware_only_correct=n10, blind_only_correct=n01, p_mcnemar=p))
TS = pd.DataFrame(tests); TS.to_csv(RES / "exp_a_tests.csv", index=False)

# per-library stale rate (blind, heading queries, changed sections)
LB = A[(A["mode"] == "blind") & (A.cls == "changed") & (A.qtype == "heading")].groupby(["lib", "retriever"]).stale1.mean().unstack()
LB.to_csv(RES / "exp_b_stale_by_library.csv")

# sensitivity of the headline to the "changed" threshold
sens = []
for thr in (0.6, 0.7, 0.8):
    g = A[(A["mode"] == "blind") & (A.qtype == "heading") & (A.retriever == "bm25") & (A.jaccard < thr)]
    sens.append(dict(threshold=thr, n=len(g), hit1=g.hit1.mean(), stale1=g.stale1.mean()))
pd.DataFrame(sens).to_csv(RES / "exp_a_threshold_sensitivity.csv", index=False)

# ------------------------------------------------------------------ Exp C
new_by_lib = {l: [s for s in secs if s["lib"] == l and s["role"] == "new"] for l in LIBS}
allpairs = {l: [p for p in pairs if p["lib"] == l] for l in LIBS}
rs = np.random.default_rng(SEED)
N_Q = 150


def make_queries(index_libs, out_lib, n_ans, n_unans):
    ans = []
    for l in index_libs:
        pp = [allpairs[l][i] for i in rs.choice(len(allpairs[l]), min(n_ans, len(allpairs[l])), replace=False)]
        for p in pp:  # question phrased from the OLD docs; answer lives in the NEW docs
            o = S[p["old_id"]]; ans.append(o["heading"] + " " + first_sentence(o["text"]))
    un = [new_by_lib[out_lib][i] for i in rs.choice(len(new_by_lib[out_lib]), n_unans, replace=False)]
    un = [s["heading"] + " " + first_sentence(s["text"]) for s in un]
    return ans, un


def top_scores(index_libs, queries):
    docs = [s for l in index_libs for s in new_by_lib[l]]
    R = Retriever("tfidf", [s["heading"] + " " + s["text"] for s in docs])
    return R.scores(queries).max(1)


def best_threshold(scores, abstain_label):   # abstain if score < thr ; maximise F1 of abstain class
    best, bt = -1, 0
    for t in np.unique(scores):
        pred = scores < t
        tp = (pred & abstain_label).sum(); fp = (pred & ~abstain_label).sum(); fn = (~pred & abstain_label).sum()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best: best, bt = f1, t
    return bt


fold_rows, pooled_s, pooled_y = [], [], []
for h in LIBS:
    inc = [l for l in LIBS if l != h]
    ds, dy = [], []
    for g in inc:   # dev: index = libs without {h,g}; g plays the unanswerable library
        idx = [l for l in inc if l != g]
        a, u = make_queries(idx, g, N_Q // 2, N_Q * 2 // 2)
        ds += list(top_scores(idx, a + u)); dy += [False] * len(a) + [True] * len(u)
    thr = best_threshold(np.array(ds), np.array(dy))
    a, u = make_queries(inc, h, N_Q // 3, N_Q)
    s = top_scores(inc, a + u); y = np.array([False] * len(a) + [True] * len(u))
    pred = s < thr
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum()); tn = int((~pred & ~y).sum())
    fold_rows.append(dict(heldout=h, threshold=thr, n_ans=len(a), n_unans=len(u), tp=tp, fp=fp, fn=fn, tn=tn,
                          precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1), f1=2 * tp / max(2 * tp + fp + fn, 1),
                          answer_rate_on_answerable=tn / max(tn + fp, 1), auroc=roc_auc_score(y, -s)))
    pooled_s += list(s); pooled_y += list(y)
FC = pd.DataFrame(fold_rows); FC.to_csv(RES / "exp_c_folds.csv", index=False)
ps, py = np.array(pooled_s), np.array(pooled_y)
tot = FC[["tp", "fp", "fn", "tn"]].sum()
C_pool = dict(precision=tot.tp / (tot.tp + tot.fp), recall=tot.tp / (tot.tp + tot.fn), f1=2 * tot.tp / (2 * tot.tp + tot.fp + tot.fn),
              auroc=roc_auc_score(py, -ps), n=len(py), answerable_kept=tot.tn / (tot.tn + tot.fp))
pd.DataFrame([C_pool]).to_csv(RES / "exp_c_pooled.csv", index=False)

# ------------------------------------------------------------------ figures
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
RK = ["bm25", "tfidf", "lsa"]; RKN = {"bm25": "BM25", "tfidf": "TF-IDF", "lsa": "LSA"}
C_BLIND, C_AWARE = "#9aa5b1", "#1f5fa8"

# Fig 1: strict Hit@1, changed sections, blind vs aware, two query types
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
for ax, qt in zip(axes, ("heading", "heading+sentence")):
    for j, (md, col) in enumerate((("blind", C_BLIND), ("aware", C_AWARE))):
        vals = [SM[(SM.cls == "changed") & (SM.qtype == qt) & (SM.retriever == r) & (SM["mode"] == md)].iloc[0] for r in RK]
        y = [v.hit1 * 100 for v in vals]; err = [[(v.hit1 - v.hit1_lo) * 100 for v in vals], [(v.hit1_hi - v.hit1) * 100 for v in vals]]
        ax.bar(np.arange(3) + j * 0.38, y, 0.38, yerr=err, capsize=2, color=col, label="version-" + md)
    ax.set_xticks(np.arange(3) + 0.19); ax.set_xticklabels([RKN[r] for r in RK]); ax.set_title(f"query = {qt}", fontsize=9); ax.set_ylim(0, 105)
axes[0].set_ylabel("Hit@1 on changed sections (%)"); axes[0].legend(frameon=False, loc="upper left", fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "fig1_hit_at_1.png"); plt.close(fig)

# Fig 2: stale-hit rate by library (blind, heading queries)
fig, ax = plt.subplots(figsize=(5.4, 3.0)); w = 0.26
for j, r in enumerate(RK):
    ax.bar(np.arange(4) + j * w, [LB.loc[l, r] * 100 for l in LIBS], w, label=RKN[r], color=["#3b6ea5", "#7fa7d1", "#c4d6ea"][j])
ax.set_xticks(np.arange(4) + w); ax.set_xticklabels([LIBNAME[l] for l in LIBS]); ax.set_ylabel("Stale hit @1 (%)")
ax.set_title("Version-blind retrieval returns the other, materially different version", fontsize=8.5); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "fig2_stale_by_library.png"); plt.close(fig)

# Fig 3: ROC (pooled) + per-fold F1
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
fpr, tpr, _ = roc_curve(py, -ps); axes[0].plot(fpr, tpr, color=C_AWARE); axes[0].plot([0, 1], [0, 1], "--", color="#aaa", lw=0.8)
axes[0].set_xlabel("False positive rate (answerable query refused)"); axes[0].set_ylabel("True positive rate (unanswerable refused)")
axes[0].set_title(f"Abstention ROC, pooled (AUROC {C_pool['auroc']:.2f})", fontsize=9)
axes[1].bar([LIBNAME[h] for h in FC.heldout], FC.f1 * 100, color=C_AWARE); axes[1].set_ylabel("F1 of abstain class (%)"); axes[1].set_ylim(0, 100)
axes[1].set_title("Held-out library (threshold from other libraries)", fontsize=9); plt.setp(axes[1].get_xticklabels(), rotation=15)
fig.tight_layout(); fig.savefig(FIG / "fig3_abstention.png"); plt.close(fig)

# Fig 4: composition
comp = pd.DataFrame(pairs).groupby(["lib", "cls"]).size().unstack(fill_value=0).reindex(LIBS)
fig, ax = plt.subplots(figsize=(5.4, 2.8)); bot = np.zeros(4)
for c, col in (("unchanged", "#c4d6ea"), ("partial", "#9aa5b1"), ("changed", "#1f5fa8")):
    ax.bar([LIBNAME[l] for l in LIBS], comp[c], bottom=bot, color=col, label=c); bot += comp[c].values
ax.set_ylabel("Matched sections"); ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(FIG / "fig4_composition.png"); plt.close(fig)

# ------------------------------------------------------------------ macros + tables for the paper
def g1(cls, qt, rk, md, m): return SM[(SM.cls == cls) & (SM.qtype == qt) & (SM.retriever == rk) & (SM["mode"] == md)].iloc[0][m]
nch = int((pd.DataFrame(pairs).cls == "changed").sum()); nun = int((pd.DataFrame(pairs).cls == "unchanged").sum())
npart = int((pd.DataFrame(pairs).cls == "partial").sum())
mac = {"NSections": len(secs), "NPairs": len(pairs), "NChanged": nch, "NUnchanged": nun, "NPartial": npart,
       "NQueriesChanged": int(g1("changed", "heading", "bm25", "blind", "n")),
       "CBlindHeading": g1("changed", "heading", "bm25", "blind", "hit1") * 100, "CAwareHeading": g1("changed", "heading", "bm25", "aware", "hit1") * 100,
       "CStaleHeading": g1("changed", "heading", "bm25", "blind", "stale1") * 100,
       "CBlindFull": g1("changed", "heading+sentence", "bm25", "blind", "hit1") * 100, "CAwareFull": g1("changed", "heading+sentence", "bm25", "aware", "hit1") * 100,
       "CStaleFull": g1("changed", "heading+sentence", "bm25", "blind", "stale1") * 100,
       "UBlindEquiv": g1("unchanged", "heading", "bm25", "blind", "equiv1") * 100, "UBlindStrict": g1("unchanged", "heading", "bm25", "blind", "hit1") * 100,
       "CauROC": C_pool["auroc"], "CFscore": C_pool["f1"] * 100, "CPrec": C_pool["precision"] * 100, "CRec": C_pool["recall"] * 100,
       "CKept": C_pool["answerable_kept"] * 100, "CN": C_pool["n"]}
def rng_(qt, md, m, cls="changed"):
    v = [g1(cls, qt, rk, md, m) * 100 for rk in RK]; return min(v), max(v)
extra = {}
for qt, tag in (("heading", "H"), ("heading+sentence", "F")):
    for md, mt in (("blind", "Blind"), ("aware", "Aware")):
        lo, hi = rng_(qt, md, "hit1"); extra[f"Hit{mt}{tag}Min"], extra[f"Hit{mt}{tag}Max"] = lo, hi
    lo, hi = rng_(qt, "blind", "stale1"); extra[f"Stale{tag}Min"], extra[f"Stale{tag}Max"] = lo, hi
lo, hi = rng_("heading", "blind", "wrongver1", "unchanged"); extra["UWrongMin"], extra["UWrongMax"] = lo, hi
lo, hi = rng_("heading", "blind", "equiv1", "unchanged"); extra["UEqMin"], extra["UEqMax"] = lo, hi
lo, hi = rng_("heading", "blind", "hit1", "unchanged"); extra["UStrMin"], extra["UStrMax"] = lo, hi
lo, hi = rng_("heading", "aware", "hit1", "unchanged"); extra["UAwareMin"], extra["UAwareMax"] = lo, hi
lo, hi = rng_("heading", "blind", "rec5"); extra["RecBlindMin"], extra["RecBlindMax"] = lo, hi
lo, hi = rng_("heading", "aware", "rec5"); extra["RecAwareMin"], extra["RecAwareMax"] = lo, hi
extra["SensMin"], extra["SensMax"] = pd.DataFrame(sens).stale1.min() * 100, pd.DataFrame(sens).stale1.max() * 100
extra["SensNLow"] = int(pd.DataFrame(sens).n.min())
extra["LibStaleMin"], extra["LibStaleMax"] = LB.values.min() * 100, LB.values.max() * 100
fb = FC.loc[FC.f1.idxmin()]; fg = FC.loc[FC.f1.idxmax()]
extra["FoldLowF"], extra["FoldHighF"] = fb.f1 * 100, fg.f1 * 100
extra["FoldLowAU"], extra["FoldHighAU"] = FC.auroc.min(), FC.auroc.max()
macros_text = {"FoldLowName": LIBNAME[fb.heldout], "FoldHighName": LIBNAME[fg.heldout]}
mac.update(extra)
with open(RES / "macros.tex", "w") as f:
    for k, v in mac.items():
        if k == "CauROC": f.write("\\newcommand{\\CauROC}{%.3f}\n" % C_pool["auroc"])
        elif k.startswith(("FoldLowAU", "FoldHighAU")): f.write("\\newcommand{\\%s}{%.3f}\n" % (k, v))
        elif isinstance(v, float): f.write(f"\\newcommand{{\\{k}}}{{{v:.1f}}}\n")
        else: f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    for k, v in macros_text.items(): f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

def fmt(r, m, s=100, d=1): return f"{r[m]*s:.{d}f} [{r[m+'_lo']*s:.{d}f}, {r[m+'_hi']*s:.{d}f}]"
L = [r"\begin{tabular}{llccccc}", r"\toprule", r"Query & Retriever & Mode & Hit@1 & Stale@1 & Recall@5 & MRR \\", r"\midrule"]
for qt in ("heading", "heading+sentence"):
    for rk in RK:
        for md in ("blind", "aware"):
            r = SM[(SM.cls == "changed") & (SM.qtype == qt) & (SM.retriever == rk) & (SM["mode"] == md)].iloc[0]
            L.append(f"{qt.replace('+', '+')} & {RKN[rk]} & {md} & {fmt(r,'hit1')} & {fmt(r,'stale1')} & {fmt(r,'rec5')} & {r.rr:.3f} \\\\")
    if qt == "heading": L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}"]
(RES / "table_exp_ab.tex").write_text("\n".join(L) + "\n")

L = [r"\begin{tabular}{llccc}", r"\toprule", r"Retriever & Mode & Strict Hit@1 & Equivalent Hit@1 & Wrong version@1 \\", r"\midrule"]
for rk in RK:
    for md in ("blind", "aware"):
        r = SM[(SM.cls == "unchanged") & (SM.qtype == "heading") & (SM.retriever == rk) & (SM["mode"] == md)].iloc[0]
        L.append(f"{RKN[rk]} & {md} & {fmt(r,'hit1')} & {fmt(r,'equiv1')} & {fmt(r,'wrongver1')} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
(RES / "table_unchanged.tex").write_text("\n".join(L) + "\n")

L = [r"\begin{tabular}{lcccccc}", r"\toprule", r"Held-out library & Threshold & Precision & Recall & F1 & AUROC & Answerable kept \\", r"\midrule"]
for _, r in FC.iterrows():
    L.append(f"{LIBNAME[r.heldout]} & {r.threshold:.3f} & {r.precision*100:.1f} & {r.recall*100:.1f} & {r.f1*100:.1f} & {r.auroc:.2f} & {r.answer_rate_on_answerable*100:.1f} \\\\")
L += [r"\midrule", f"Pooled & -- & {C_pool['precision']*100:.1f} & {C_pool['recall']*100:.1f} & {C_pool['f1']*100:.1f} & {C_pool['auroc']:.2f} & {C_pool['answerable_kept']*100:.1f} \\\\",
      r"\bottomrule", r"\end{tabular}"]
(RES / "table_exp_c.tex").write_text("\n".join(L) + "\n")


L = [r"\begin{tabular}{llccc}", r"\toprule", r"Query & Retriever & Aware-only correct & Blind-only correct & $p$ \\", r"\midrule"]
for _, r in TS[TS.cls == "changed"].iterrows():
    pv = "$<$0.001" if r.p_mcnemar < 0.001 else f"{r.p_mcnemar:.3f}"
    L.append(f"{r.qtype} & {RKN[r.retriever]} & {r.aware_only_correct} & {r.blind_only_correct} & {pv} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
(RES / "table_mcnemar.tex").write_text("\n".join(L) + "\n")


# ---- appendix material: real changed-section examples and per-library table
def tex_esc(s):
    s = s.replace("\\", " ")
    for a, b in (("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"), ("_", "\\_"), ("{", "\\{"), ("}", "\\}"), ("~", " "), ("^", " ")):
        s = s.replace(a, b)
    return s
ex = []
for lib in LIBS:
    cands = [q for q in pairs if q["lib"] == lib and q["cls"] == "changed" and len(S[q["new_id"]]["heading"]) <= 45]
    q = sorted(cands, key=lambda z: (abs(z["jaccard"] - 0.5), z["key"]))[0]
    o, n = S[q["old_id"]], S[q["new_id"]]
    cut = lambda s: " ".join(s["text"].split()[:45])
    ex.append(f"\\paragraph{{{LIBNAME[lib]}: ``{tex_esc(n['heading'])}'' (Jaccard {q['jaccard']:.2f}).}} "
              f"\\emph{{{tex_esc(o['version'])}:}} {tex_esc(cut(o))}\\,\\dots{{}} \\emph{{{tex_esc(n['version'])}:}} {tex_esc(cut(n))}\\,\\dots\n")
(RES / "examples.tex").write_text("\n".join(ex))
L = [r"\begin{tabular}{lccc}", r"\toprule", r"Library & BM25 & TF-IDF & LSA \\", r"\midrule"]
for l in LIBS:
    L.append(f"{LIBNAME[l]} & {LB.loc[l,'bm25']*100:.1f} & {LB.loc[l,'tfidf']*100:.1f} & {LB.loc[l,'lsa']*100:.1f} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
(RES / "table_by_library.tex").write_text("\n".join(L) + "\n")

pd.set_option("display.width", 200)
print(SM[(SM.cls == "changed")][["qtype", "retriever", "mode", "n", "hit1", "stale1", "rec5", "rr"]].round(3).to_string(index=False))
print("\nunchanged:\n", SM[(SM.cls == "unchanged") & (SM.qtype == "heading")][["retriever", "mode", "n", "hit1", "equiv1", "wrongver1"]].round(3).to_string(index=False))
print("\nstale by lib:\n", LB.round(3)); print("\nsensitivity:\n", pd.DataFrame(sens).round(3)); print("\nExp C folds:\n", FC.round(3)); print(C_pool)
print("\nMcNemar:\n", TS[TS.cls == "changed"].round(4).to_string(index=False))
