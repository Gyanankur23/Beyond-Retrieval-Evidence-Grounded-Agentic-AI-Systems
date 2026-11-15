"""Write README.md from the result files, so the numbers in it cannot drift from the data."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
SM = pd.read_csv(R / "exp_ab_summary.csv"); FC = pd.read_csv(R / "exp_c_folds.csv"); CP = pd.read_csv(R / "exp_c_pooled.csv").iloc[0]
N = {"bm25": "BM25", "tfidf": "TF-IDF", "lsa": "LSA"}
LN = {"fastapi": "FastAPI", "pandas": "pandas", "sklearn": "scikit-learn", "pydantic": "Pydantic"}


def g(cls, qt, rk, md, m):
    return SM[(SM.cls == cls) & (SM.qtype == qt) & (SM.retriever == rk) & (SM["mode"] == md)].iloc[0][m]


rows = ["| Retriever | Query | Hit@1 blind | Hit@1 version-aware | Stale@1 blind | Stale@1 aware |", "|---|---|---|---|---|---|"]
for qt in ("heading", "heading+sentence"):
    for rk in N:
        rows.append(f"| {N[rk]} | {qt} | {g('changed',qt,rk,'blind','hit1')*100:.1f}% | **{g('changed',qt,rk,'aware','hit1')*100:.1f}%** | "
                    f"{g('changed',qt,rk,'blind','stale1')*100:.1f}% | {g('changed',qt,rk,'aware','stale1')*100:.1f}% |")
fold = ["| Held-out library | F1 | Precision | Recall | AUROC |", "|---|---|---|---|---|"]
for _, r in FC.iterrows():
    fold.append(f"| {LN[r.heldout]} | {r.f1*100:.1f}% | {r.precision*100:.1f}% | {r.recall*100:.1f}% | {r.auroc:.3f} |")
fold.append(f"| **Pooled** | **{CP.f1*100:.1f}%** | {CP.precision*100:.1f}% | {CP.recall*100:.1f}% | {CP.auroc:.3f} |")

(ROOT / "README.md").write_text(f"""# Does the Version Matter?
### A measurement study of version-aware retrieval over real software documentation

**Authors:** Gyanankur Pratim Baruah and Anjali Tiwari, Thakur Shyamnarayan Degree College, Mumbai
**Paper:** [`paper/manuscript.pdf`](paper/manuscript.pdf)  |  **Status:** complete, fully reproducible, no external APIs

Documentation exists in many versions, and a retriever that ignores versions can hand a question-answering
assistant a page that was only correct for an older release. We measure how often that happens on **real
documentation snapshots** of FastAPI, pandas, scikit-learn and Pydantic (two release tags each), using three classical
retrievers (BM25, TF-IDF, LSA) and **no language model**. Everything below is computed by the code in this repo.

## Headline results

* **Version-blind retrieval often returns the wrong, materially different version.** On sections whose text changed
  between releases, the top result was the *other* version for {SM[(SM.cls=='changed')&(SM.qtype=='heading')&(SM['mode']=='blind')].stale1.min()*100:.0f}-{SM[(SM.cls=='changed')&(SM.qtype=='heading')&(SM['mode']=='blind')].stale1.max()*100:.0f}% of topic-style (heading) queries.
* **Filtering by the requested version removes that error completely** and raises top-1 accuracy from
  {SM[(SM.cls=='changed')&(SM.qtype=='heading')&(SM['mode']=='blind')].hit1.min()*100:.0f}-{SM[(SM.cls=='changed')&(SM.qtype=='heading')&(SM['mode']=='blind')].hit1.max()*100:.0f}% to {SM[(SM.cls=='changed')&(SM.qtype=='heading')&(SM['mode']=='aware')].hit1.min()*100:.0f}-{SM[(SM.cls=='changed')&(SM.qtype=='heading')&(SM['mode']=='aware')].hit1.max()*100:.0f}%.
* **A single retrieval-score threshold can refuse most out-of-corpus questions**: pooled leave-one-library-out F1 of
  {CP.f1*100:.1f}% (AUROC {CP.auroc:.3f}) while still answering {CP.answerable_kept*100:.1f}% of answerable queries.

### Experiment A/B: version confusion on changed sections (95% CIs are in the paper)
{chr(10).join(rows)}

*Stale@1 = the top result is the other version's section and its content differs materially. Aware mode searches only the requested version, so it cannot return a stale hit by construction.*

![Hit@1 on changed sections](results/figures/fig1_hit_at_1.png)

![Stale hit rate by library](results/figures/fig2_stale_by_library.png)

### Experiment C: abstaining on out-of-corpus queries (leave-one-library-out)
{chr(10).join(fold)}

![Abstention ROC and per-library F1](results/figures/fig3_abstention.png)

### Data composition
![Matched sections per library and change class](results/figures/fig4_composition.png)

## Reproduce (about a minute, CPU only, no network)
```bash
pip install -r requirements.txt
./reproduce.sh        # parse docs -> run study -> regenerate README -> rebuild paper PDF (needs pdflatex)
```
Everything is deterministic (seed 0). `scripts/fetch_docs.sh` re-downloads the snapshots at the pinned tags if you want to verify them.

## Layout
```
data/raw_docs/          documentation snapshots (md/rst only; upstream licences, see data/NOTICE.md)
data/sections.jsonl     parsed sections;   data/pairs.json  version-paired sections with change class
scripts/                fetch_docs.sh, build_sections.py, make_readme.py
experiments/run_study.py   all experiments, figures, tables and paper macros
results/                CSVs, per-query results, figures/, LaTeX tables and macros.tex
paper/                  manuscript.tex / .pdf, references.bib, build.sh
```

## Limitations (details in the paper)
Synthetic queries built from headings; section-level relevance; "changed" is a lexical definition; classical retrievers
only (no trained dense models); the abstention task uses easy out-of-library queries; four libraries, two releases each;
no language model is evaluated. This is a small measurement study, not a benchmark of question-answering systems.

## Cite
See `CITATION.cff`.
""")
print("README.md written")
