# Reproducibility Guide: Agentic RAG Reliability Study

This guide provides step-by-step instructions for setting up the environment, executing experimental sweeps, running statistical analysis, and compiling the manuscript to fully reproduce the findings in *"Beyond Retrieval: Evaluating Evidence-Grounded Agentic AI Systems for Reliable Technical Question Answering"*.

---

## 1. Environment Setup

### Prerequisites
* Python 3.10+
* Git
* LaTeX distribution (TeX Live / MacTeX) for PDF compilation

### Installation Steps
Clone the repository and install the required Python dependencies:

```bash
git clone https://github.com/Gyanankur23/Beyond-Retrieval-Evidence-Grounded-Agentic-AI-Systems.git
cd Beyond-Retrieval-Evidence-Grounded-Agentic-AI-Systems
pip install -r requirements.txt
```

---

## 2. Dataset Verification

Verify that the benchmark dataset and literature review assets are correctly placed:
* `data/questions.json` (100 benchmark questions)
* `literature_review/search_protocol.md` 
* `literature_review/screening_log.csv` 
* `literature_review/literature_matrix.csv` 

Run the setup validation script to test imports and dataset parsing:
```bash
python verify_setup.py
```

---

## 3. Build the corpus (required, not shipped)

`data/questions.json` references 157 source files that must exist in `data/processed/`.
1. `python scripts/make_manifest.py` writes `data/corpus_manifest.csv` (filenames + *proposed* version/status).
2. For each row, save the real upstream document snapshot to `data/processed/<filename>`, and fill
   `upstream_url`, `retrieved_on`, and confirm `doc_version`/`status`; then set `confirmed=yes`.
   Paired "outdated" and "contradictory" files must be genuine versions/sources, not authored stand-ins.
3. The runner refuses to start until every referenced file exists and is confirmed.

## 4. Live evaluation

```bash
export ANTHROPIC_API_KEY=...
python experiments/run_evaluations.py --config all \
    --model <generator-model> --judge-model <different-judge-model> \
    --price-in <USD per M input tokens> --price-out <USD per M output tokens>
```
Writes `results/raw_results/evaluation_log.csv` (400 rows for E1-E4; 800 with `--config all`) and
`traces.jsonl` (every prompt-level output, for audit). `--dry-run --backend tfidf` tests the plumbing
with a stub model and writes only to `results/dry_run/`.

**Judge validation (required before reporting accuracy):**
`python -m src.live.judge_validation export` -> both authors label `human_a`/`human_b` independently ->
`python -m src.live.judge_validation score` reports Cohen's kappa (A vs B, judge vs consensus).

## 5. Statistical analysis and table

```bash
python results/statistical_analysis/analyze_results.py   # McNemar / Wilcoxon, bootstrap CIs, Holm
python paper/make_table.py                                # paper/tables/results_table.tex
```
The analysis stops if the log is missing or contains dry-run rows.

## 6. Compiling the Research Manuscript

To compile the LaTeX source code into the final publication-ready PDF:

```bash
cd paper/
pdflatex manuscript.tex
bibtex manuscript
pdflatex manuscript.tex
pdflatex manuscript.tex
cd ..
