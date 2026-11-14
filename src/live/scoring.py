"""Scoring.

answer_accuracy
  insufficient_evidence items : 1 iff the system abstained (an answer is a hallucination).
  all other items             : 0 if the system abstained; otherwise an LLM judge compares the
                                answer with ground_truth (judge model should differ from the
                                generator; agreement with human labels must be reported,
                                see judge_validation.py).
citation_precision
  |cited ∩ target_sources| / |cited|, computed at document level. Undefined (NaN) when the system
  cites nothing (e.g. abstains); the analysis reports the number of defined items per system.
"""
from .llm import parse_json

JUDGE_SYS = (
    "You are a strict judge of technical answers. Compare the candidate answer with the reference "
    "answer. The candidate is correct only if it conveys the same technical facts as the reference "
    "and does not contradict it; extra correct detail is fine, missing or wrong key facts are not. "
    'JSON only: {"verdict": "correct"|"incorrect", "reason": str}'
)


def judge(judge_llm, question: str, reference: str, candidate: str) -> int:
    r = parse_json(judge_llm.call(JUDGE_SYS,
        f"Question: {question}\nReference answer: {reference}\nCandidate answer: {candidate}", 200))
    return int(str(r.get("verdict", "")).lower() == "correct")


def score_accuracy(judge_llm, q: dict, out) -> int:
    if q["category"] == "insufficient_evidence":
        return int(out.abstained)
    if out.abstained:
        return 0
    return judge(judge_llm, q["question"], q["ground_truth"], out.answer)


def citation_precision(cited_files: list, targets: list):
    if not cited_files:
        return float("nan")
    return len(set(cited_files) & set(targets)) / len(set(cited_files))
