# Technical Documentation QA Benchmark Dataset
**Dataset File:** `data/questions.json`  
**Total Records:** 100 validated technical questions  
**License:** CC BY 4.0 (Original annotations and question formulations; underlying third-party technical documentation retains original copyright and licensing)

## 1. Overview & Purpose
This dataset is constructed to evaluate Retrieval-Augmented Generation (RAG) and evidence-grounded multi-agent architectures under controlled conditions of technical complexity, version drift, conflicting guidance, and missing evidence. Unlike general-purpose QA datasets, every item requires exact adherence to verifiable software documentation across Python, FastAPI, LangChain/LangGraph, ChromaDB, Pandas, Pydantic, Streamlit, and GitHub.

## 2. Category Distribution
The 100 questions are evenly stratified across five evidence conditions (20 questions per category):

| Category ID | Question Count | Description | Expected System Behavior |
| :--- | :--- | :--- | :--- |
| `direct_factual` | 20 (`Q01–Q04`, `Q21–Q36`) | Single-document factual retrieval with unambiguous answers. | `direct_answer` |
| `multi_document` | 20 (`Q05–Q08`, `Q37–Q52`) | Requires synthesizing information across two or more distinct technical guides. | `multi_doc_synthesis` |
| `outdated_document` | 20 (`Q09–Q12`, `Q53–Q68`) | Pairings containing deprecated legacy APIs alongside current version specifications. | `version_resolution_override` |
| `contradictory_source` | 20 (`Q13–Q16`, `Q69–Q84`) | Sources presenting conflicting behaviors, defaults, or execution constraints depending on context. | `conflict_detection_and_flagging` |
| `insufficient_evidence` | 20 (`Q17–Q20`, `Q85–Q100`) | Unanswerable or out-of-scope queries where the corpus lacks supporting facts. | `selective_abstention` |

## 3. JSONL Schema Specification
Each line in `data/questions.json` is a self-contained JSON object adhering to the following schema:

* **`question_id`** *(string)*: Unique identifier (`Q01` through `Q100`).
* **`category`** *(string)*: One of the five stratified evaluation categories.
* **`question`** *(string)*: The technical query submitted to the evaluation pipelines.
* **`target_sources`** *(list of strings)*: Reference filenames in `data/processed/` containing the relevant evidence or conflicting context.
* **`version_context`** *(string)*: Target library version constraint (e.g., `v0.110+`, `v2.0+`, `latest`).
* **`expected_behavior`** *(string)*: Ground-truth routing/control label expected from the Response Controller.
* **`ground_truth`** *(string)*: Reference answer or explicit `ABSTAIN:` rationale verified against official documentation.

## 4. Annotation & Quality Control Protocol
1. **Source Grounding:** All ground-truth answers are derived directly from official documentation snapshots frozen prior to experimentation.
2. **Dual-Researcher Verification:** Both researchers (Gyanankur Pratim Baruah and Anjali Tiwari) independently review every question, source mapping, and expected behavior label. Disagreements on contradiction boundaries or abstention thresholds are resolved via consensus prior to running experimental sweeps.
3. **Abstention Calibration:** Questions in the `insufficient_evidence` category are specifically designed to test false-positive generation (hallucination under pressure) and must trigger an explicit refusal rather than speculative extrapolation.
