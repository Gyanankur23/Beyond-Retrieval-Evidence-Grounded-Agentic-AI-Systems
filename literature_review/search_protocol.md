# Rapid Literature Screening & Synthesis Protocol
**Project:** Beyond Retrieval: Evaluating Evidence-Grounded Agentic AI Systems for Reliable Technical Question Answering  
**Status:** Active Protocol — Version 1.0  

## 1. Objective
This protocol establishes a standardized, reproducible methodology for conducting a targeted rapid literature review across major academic preprints and peer-reviewed repositories. The review aims to map existing work in agentic Retrieval-Augmented Generation (RAG), citation verification, document-version awareness, contradiction detection, and selective abstention.

## 2. Target Databases
The literature search will be executed across the following digital libraries:
1. **arXiv** (Computer Science / Computation and Language preprints)
2. **Google Scholar** (Broad multidisciplinary academic discovery)
3. **ACL Anthology** (Computational linguistics and NLP venues)
4. **ACM Digital Library** (Agent systems and software engineering)
5. **IEEE Xplore** (AI reliability and engineering applications)
6. **Semantic Scholar** (Citation graph analysis and related paper discovery)

## 3. Standardized Search Strings
The following ten independent search strings will be queried without unnecessary boolean bloating:

- **Q1:** `"retrieval augmented generation" AND "citation verification"` 
- **Q2:** `"claim-level attribution" AND "large language models"` 
- **Q3:** `"multi-agent RAG" AND "evidence verification"` 
- **Q4:** `"agentic RAG" AND "reliability evaluation"` 
- **Q5:** `"version-aware retrieval" OR "version-aware RAG"` 
- **Q6:** `"contradictory evidence" AND "retrieval augmented generation"` 
- **Q7:** `"evidence sufficiency" AND "answer abstention" AND LLM` 
- **Q8:** `"selective prediction" AND "retrieval augmented generation"` 
- **Q9:** `"multi-agent systems" AND "computational cost" AND LLM` 
- **Q10:** `"technical documentation" AND "evidence-grounded question answering"` 

## 4. Inclusion Criteria
A publication is included if it meets at least one of the following criteria:
- Directly investigates RAG reliability, failure modes, or citation faithfulness.
- Proposes or evaluates multi-agent RAG architectures.
- Studies claim-level evidence attribution or verification mechanisms.
- Evaluates abstention calibration, uncertainty estimation, or refusal behavior.
- Investigates version-sensitive document retrieval (e.g., VersionRAG).
- Presents reproducible technical benchmarks or evaluation frameworks.

## 5. Exclusion Criteria
A publication is excluded if:
- It lacks any substantive connection to the core research questions.
- It consists solely of commercial vendor marketing, product blogs, or promotional materials.
- It is an unverified duplicate version of a study (while retaining primary publication history).
- It lacks sufficient methodological detail to evaluate experimental validity.
- Full text or implementation details are entirely inaccessible.

## 6. Screening Workflow
1. **Execution & Logging:** Execute search queries, record retrieval counts, and store records in `screening_log.csv`.
2. **Deduplication:** Remove duplicate entries across databases while preserving distinct preprint vs. conference publication records.
3. **Title & Abstract Screening:** Independently screen titles and abstracts against inclusion/exclusion criteria.
4. **Full-Text Assessment:** Review full texts of shortlisted papers, extract core comparative findings into `literature_matrix.csv`, and document exclusion rationales.
5. **Citation Chaining:** Perform backward and forward citation searches on key papers (e.g., VersionRAG, TechDocRAG) to capture recent 2025–2026 developments.
