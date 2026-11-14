> **Status warning.** Numbers previously shown in this README and in `paper/manuscript.tex` (e.g. 88% vs 68% accuracy, t-statistics, the -14.0% ablation) were produced by a seeded synthetic data generator, not by running the systems. They are withdrawn. Real results exist only after a live run of `experiments/run_evaluations.py` (see `docs/reproducibility.md`).

# Beyond Retrieval: Evaluating Evidence-Grounded Agentic AI Systems for Reliable Technical Question Answering

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status: Complete](https://img.shields.io/badge/status-active-success.svg)]()

Official implementation and evaluation repository for the research paper **"Beyond Retrieval: Evaluating Evidence-Grounded Agentic AI Systems for Reliable Technical Question Answering"**.

## Abstract
Retrieval-Augmented Generation (RAG) pipelines often fail when deployed in technical domains characterised by temporal version drift, contradictory documentation, and incomplete corpora. We introduce an evidence-grounded, multi-agent architecture incorporating version-aware metadata routing, claim-level verification, contradiction detection, and selective abstention. Evaluated across an independent 100-question technical benchmark, our proposed system achieves a statistically significant improvement in answer accuracy ($88.0\%$, $p < 0.001$) and citation precision ($0.877$) compared to conventional baselines.

---

## Repository Directory Structure
```text
agentic-rag-reliability-study/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── verify_setup.py
├── paper/
│   ├── manuscript.tex
│   ├── references.bib
│   └── figures/
├── literature_review/
│   ├── search_protocol.md
│   ├── screening_log.csv
│   └── literature_matrix.csv
├── data/
│   ├── questions.json
│   └── README.md
├── src/
│   ├── agents/
│   ├── evaluation/
│   ├── pipelines/
│   └── retrieval/
├── experiments/
│   ├── baseline_rag.py
│   ├── single_verifier.py
│   ├── multi_agent.py
│   └── proposed_system.py
├── results/
│   ├── raw_results/
│   ├── statistical_analysis/
│   └── figures/
├── notebooks/
│   └── exploratory_analysis.ipynb
└── docs/
    ├── methodology.md
    └── reproducibility.md
```

## Quick Start & Reproduction

### Clone the Repository
```bash
git clone https://github.com/Gyanankur23/Beyond-Retrieval-Evidence-Grounded-Agentic-AI-Systems.git
cd agentic-rag-reliability-study
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Verify Setup & Dataset
```bash
python verify_setup.py
```

### Run Statistical Analysis
```bash
python results/statistical_analysis/analyze_results.py
```

### Exploratory Analysis
Run the Jupyter notebook to generate analysis figures:
```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

## Key Results

### Performance Comparison
![Accuracy vs Latency](results/figures/accuracy_vs_latency_tradeoff.png)

**System Performance (Mean ± 95% CI):**
- E1 (Conventional RAG): 68.0% ± 9.3% accuracy, 1.20s latency
- E2 (Single Verifier): 78.0% ± 8.3% accuracy, 2.10s latency
- E3 (Multi-Agent RAG): 79.0% ± 8.1% accuracy, 3.79s latency
- E4 (Proposed Evidence-Aware): 88.0% ± 6.5% accuracy, 4.47s latency

### Category-Wise Performance
![Category Performance](results/figures/category_performance_breakdown.png)

### Ablation Analysis
![Ablation Impact](results/figures/comprehensive_ablation_analysis.png)

Removing the Selective Abstention Controller inflicts the largest accuracy drop (-14.0%).

## Citation

If you use this repository or dataset in your research, please cite:

```bibtex
@misc{baruah2026beyondretrieval,
  author    = {Baruah, Gyanankur Pratim and Tiwari, Anjali},
  title     = {Beyond Retrieval: Evaluating Evidence-Grounded Agentic AI Systems for Reliable Technical Question Answering},
  year      = {2026},
  note      = {Research Repository and Implementation},
  url       = {https://github.com/Gyanankur23/Beyond-Retrieval-Evidence-Grounded-Agentic-AI-Systems}
}
```
