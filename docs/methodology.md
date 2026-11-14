# Methodology Documentation: Experimental Framework

This document outlines the theoretical and architectural design behind the evaluation framework for *"Beyond Retrieval: Evaluating Evidence-Grounded Agentic AI Systems for Reliable Technical Question Answering"*.

---

## 1. Research Objectives
The study investigates whether explicit multi-agent verification, temporal version routing, contradiction handling, and selective abstention improve technical question answering reliability compared to conventional Retrieval-Augmented Generation (RAG).

---

## 2. Experimental Configurations (E1–E4)

1. **Configuration E1 (Conventional RAG):** Baseline system performing standard top-$k$ vector retrieval followed by direct LLM generation without verification or filtering.
2. **Configuration E2 (RAG + Single Verifier):** Standard retrieval and generation augmented by a post-hoc claim verification agent.
3. **Configuration E3 (Multi-Agent RAG):** Task-decomposed architecture utilizing a Query Analyst, Retrieval Agent, Evidence Analyst, and Response Controller.
4. **Configuration E4 (Proposed Evidence-Aware System):** Full multi-agent orchestration integrating version-context metadata filtering, explicit Conflict Detection, claim-level Verifier, and a Selective Abstention Controller (`ABSTAIN:`).

---

## 3. Benchmark Stratification (`data/questions.json`)
The 100-question dataset is evenly distributed across five adversarial conditions ($n=20$ each):
* `direct_factual`: Single-document unambiguous lookups.
* `multi_document`: Dependent multi-source synthesis.
* `outdated_document`: Legacy vs. current API version resolution.
* `contradictory_source`: Conflicting documentation handling.
* `insufficient_evidence`: Out-of-scope queries testing calibration and selective abstention.

---

## 4. Evaluation Metrics
* **Epistemic Reliability:** Answer accuracy (binary ground-truth match), citation precision, contradiction detection F1, and abstention precision/recall.
* **Operational Overhead:** End-to-end latency (seconds) and estimated token cost (USD).
* **Statistical Rigor:** Paired Student's t-tests ($\alpha = 0.05$) and 95% confidence intervals derived from t-distributions.
