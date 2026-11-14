"""Pipelines E1-E4 (+ E4 ablations) over the live index and LLM.

Inputs available to every system: question text and the target version string.
NOT available to any system: category, expected_behavior, target_sources, ground_truth.
All systems share the same answer prompt, including an explicit option to abstain, so the
baselines are not handicapped by a prompt that forbids refusal.
"""
import re
from dataclasses import dataclass, field

from .llm import parse_json

ABSTAIN_TAU = 0.70   # E4 claim-support threshold, fixed a priori (not tuned on the benchmark)

ANSWER_SYS = (
    "You answer technical questions about software libraries using ONLY the documentation "
    "excerpts provided. Each excerpt is tagged [DOC#chunk]. Cite the DOC ids you rely on. "
    "If the excerpts do not contain enough information to answer, set abstain to true and "
    'leave answer empty. Reply with JSON only: {"answer": str, "citations": [str], "abstain": bool}'
)


@dataclass
class Output:
    answer: str = ""
    citations: list = field(default_factory=list)   # opaque doc ids
    abstained: bool = False
    trace: dict = field(default_factory=dict)


def _docs_of(refs) -> list:
    seen, out = set(), []
    for r in refs or []:
        d = str(r).strip("[] ").split("#")[0]
        if re.fullmatch(r"D\d{3}", d) and d not in seen:
            seen.add(d); out.append(d)
    return out


def _fmt(chunks, with_meta=False) -> str:
    parts = []
    for c in chunks:
        tag = f"[{c.ref}]"
        if with_meta:
            tag += f" (library={c.meta['library']}; doc_version={c.meta['doc_version']}; status={c.meta['status']})"
        parts.append(f"{tag}\n{c.text}")
    return "\n\n".join(parts)


def _dedupe(chunks, limit):
    seen, out = set(), []
    for c in sorted(chunks, key=lambda c: -c.score):
        if c.ref not in seen:
            seen.add(c.ref); out.append(c)
    return out[:limit]


def _generate(llm, question, version, chunks, with_meta=False, notes="") -> dict:
    user = (f"Target version: {version}\nQuestion: {question}\n\nExcerpts:\n{_fmt(chunks, with_meta)}"
            + (f"\n\n{notes}" if notes else ""))
    return parse_json(llm.call(ANSWER_SYS, user, 900))


def _as_output(d: dict, trace: dict) -> Output:
    abst = bool(d.get("abstain", False)) or not str(d.get("answer", "")).strip()
    return Output(answer="" if abst else str(d.get("answer", "")),
                  citations=[] if abst else _docs_of(d.get("citations")),
                  abstained=abst, trace=trace)


# ---------------------------------------------------------------- E1
def run_e1(llm, index, q) -> Output:
    chunks = index.search(f"{q['question']} ({q['version_context']})", 5)
    d = _generate(llm, q["question"], q["version_context"], chunks)
    return _as_output(d, {"retrieved": [c.ref for c in chunks]})


# ---------------------------------------------------------------- E2
VERIFY_SYS = (
    "You are a verifier. Given excerpts, a question and a draft answer, decide whether every "
    "statement in the draft is supported by the excerpts. Reply with JSON only: "
    '{"supported": bool, "revised_answer": str|null, "citations": [str], "abstain": bool}. '
    "If unsupported, either give a revised answer fully supported by the excerpts or set abstain true."
)


def run_e2(llm, index, q) -> Output:
    chunks = index.search(f"{q['question']} ({q['version_context']})", 5)
    d = _generate(llm, q["question"], q["version_context"], chunks)
    trace = {"retrieved": [c.ref for c in chunks], "draft": d}
    if d.get("abstain") or not str(d.get("answer", "")).strip():
        return _as_output(d, trace)
    v = parse_json(llm.call(VERIFY_SYS,
        f"Question: {q['question']}\n\nExcerpts:\n{_fmt(chunks)}\n\nDraft answer: {d.get('answer')}\n"
        f"Draft citations: {d.get('citations')}", 700))
    trace["verifier"] = v
    if v.get("supported", True):
        return _as_output(d, trace)
    if v.get("abstain") or not v.get("revised_answer"):
        return Output(abstained=True, trace=trace)
    return _as_output({"answer": v["revised_answer"], "citations": v.get("citations") or d.get("citations")}, trace)


# ---------------------------------------------------------------- E3
QA_SYS = ("You are a query analyst. Rewrite the question into at most 3 short, diverse search "
          'queries. Reply with JSON only: {"sub_queries": [str]}')
CRITIC_SYS = ("You are a critic. Review the draft answer against the excerpts and list errors, "
              'unsupported claims or omissions. Reply with JSON only: {"critique": str}')


def run_e3(llm, index, q) -> Output:
    qa = parse_json(llm.call(QA_SYS, f"Question: {q['question']}\nVersion: {q['version_context']}", 300))
    subs = [s for s in qa.get("sub_queries", []) if isinstance(s, str)][:3] or [q["question"]]
    chunks = _dedupe([c for s in subs for c in index.search(s, 4)], 6)
    d = _generate(llm, q["question"], q["version_context"], chunks)
    trace = {"sub_queries": subs, "retrieved": [c.ref for c in chunks], "draft": d}
    if d.get("abstain") or not str(d.get("answer", "")).strip():
        return _as_output(d, trace)
    cr = parse_json(llm.call(CRITIC_SYS,
        f"Question: {q['question']}\n\nExcerpts:\n{_fmt(chunks)}\n\nDraft: {d.get('answer')}", 500))
    trace["critique"] = cr
    d2 = _generate(llm, q["question"], q["version_context"], chunks,
                   notes=f"Draft answer: {d.get('answer')}\nCritic feedback: {cr.get('critique', '')}\n"
                         "Produce the final answer.")
    return _as_output(d2, trace)


# ---------------------------------------------------------------- E4
QUERY_SYS = ("You are a query analyst. Identify the library and the target version, and write at most 3 "
             'short, diverse search queries. JSON only: {"library": str, "target_version": str, "sub_queries": [str]}')
EVID_SYS = ("You are an evidence analyst. Using only the excerpts, judge whether they contain enough "
            'information to answer the question. JSON only: {"sufficiency": "sufficient"|"partial"|"insufficient", '
            '"relevant_docs": [str]}')
CONF_SYS = ("You are a conflict detector. Identify statements in the excerpts that contradict each other, or "
            "where a superseded (status=deprecated) document disagrees with a current one for the target version. "
            'JSON only: {"conflicts": [{"docs": [str], "description": str, "prefer": str|null}]}')
VER_SYS = ("You are a claim-level verifier. Split the answer into atomic claims and check each against the "
           'excerpts. JSON only: {"claims": [{"text": str, "supported": bool, "docs": [str]}]}')


def run_e4(llm, index, q, *, version=True, conflict=True, verifier=True, abstain=True) -> Output:
    qa = parse_json(llm.call(QUERY_SYS, f"Question: {q['question']}\nVersion: {q['version_context']}", 300))
    subs = [s for s in qa.get("sub_queries", []) if isinstance(s, str)][:3] or [q["question"]]
    chunks = _dedupe([c for s in subs for c in index.search(s, 4)], 8)
    if version:   # version-aware routing: current documentation first, deprecated kept but labelled
        chunks = sorted(chunks, key=lambda c: (c.meta["status"] == "deprecated", -c.score))
    chunks = chunks[:6]
    trace = {"analyst": qa, "retrieved": [c.ref for c in chunks]}

    ev = parse_json(llm.call(EVID_SYS, f"Question: {q['question']}\n\nExcerpts:\n{_fmt(chunks, version)}", 400))
    trace["evidence"] = ev
    notes = ""
    if conflict:
        cf = parse_json(llm.call(CONF_SYS, f"Target version: {q['version_context']}\n\n"
                                           f"Excerpts:\n{_fmt(chunks, True)}", 500))
        trace["conflicts"] = cf
        if cf.get("conflicts"):
            notes = f"Detected conflicts between sources: {cf['conflicts']}. Resolve for the target version or state the conflict."
    d = _generate(llm, q["question"], q["version_context"], chunks, with_meta=version, notes=notes)
    trace["draft"] = d

    if d.get("abstain") or not str(d.get("answer", "")).strip():
        return _as_output(d, trace)
    cites = _docs_of(d.get("citations"))
    if verifier:
        vr = parse_json(llm.call(VER_SYS, f"Excerpts:\n{_fmt(chunks, version)}\n\nAnswer: {d['answer']}", 700))
        trace["verification"] = vr
        claims = [c for c in vr.get("claims", []) if isinstance(c, dict)]
        if claims:
            frac = sum(bool(c.get("supported")) for c in claims) / len(claims)
            trace["supported_fraction"] = frac
            if abstain and frac < ABSTAIN_TAU:
                return Output(abstained=True, trace=trace)
            sup_docs = _docs_of([x for c in claims if c.get("supported") for x in c.get("docs", [])])
            cites = sup_docs or cites          # claim-level provenance
    if abstain and ev.get("sufficiency") == "insufficient":
        return Output(abstained=True, trace=trace)
    return Output(answer=str(d["answer"]), citations=cites, abstained=False, trace=trace)


ABLATIONS = {
    "E4_no_abstention": dict(abstain=False),
    "E4_no_conflict": dict(conflict=False),
    "E4_no_verifier": dict(verifier=False),
    "E4_no_version": dict(version=False),
}
CONFIGS = {
    "E1_Conventional_RAG": run_e1,
    "E2_Single_Verifier": run_e2,
    "E3_Multi_Agent": run_e3,
    "E4_Proposed_Evidence_Aware": run_e4,
    **{k: (lambda llm, idx, q, kw=kw: run_e4(llm, idx, q, **kw)) for k, kw in ABLATIONS.items()},
}
