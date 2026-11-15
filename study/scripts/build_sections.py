"""Parse the raw doc snapshots into sections -> data/sections.jsonl, and pair sections across versions.

A section = a heading plus the text up to the next heading. Pairs are sections whose normalised
heading is unique in both versions of a library. Change class (by word-set Jaccard of the two texts):
  unchanged >= 0.95 | changed < 0.80 | partial otherwise (excluded from the main experiments).
"""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw_docs"
VERSIONS = {"fastapi": ("0.70.0", "0.115.0"), "pandas": ("v1.5.3", "v2.2.0"),
            "sklearn": ("0.24.2", "1.4.0"), "pydantic": ("v1.10.13", "v2.5.0")}
SKIP_FILE = re.compile(r"(release|changelog|history|whatsnew|contributing|sponsor|external-links|newsletter|help)", re.I)
GENERIC = {"introduction", "example", "examples", "recap", "check it", "install", "installation", "summary",
           "conclusion", "usage", "overview", "note", "notes", "see also", "references", "tip", "warning", "alternatives",
           "technical details", "in short", "review", "next steps", "intro", "background", "limitations"}
MIN_WORDS, MAX_WORDS = 30, 400
RST_ADORN = re.compile(r'^([=\-~^"\'`#*+_.:])\1{2,}\s*$')


def clean(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)          # drop fenced code
    text = re.sub(r"^\.\. [\w:-]+::.*$", " ", text, flags=re.M)  # rst directives
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"`+|\*\*|__|:\w+:", "", text)
    text = re.sub(r"<[^>]+>|\{[^}]*\}", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def norm_head(h: str) -> str:
    h = re.sub(r"\{.*?\}|`|\*|\[|\]|\(.*?\)", " ", h.lower())
    return " ".join(re.findall(r"[a-z0-9]+", h))


def split_md(text):
    out, head, buf, fence = [], None, [], False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            fence = not fence
        m = None if fence else re.match(r"^(#{1,4})\s+(.*\S)\s*$", line)
        if m:
            if head: out.append((head, "\n".join(buf)))
            head, buf = re.sub(r"\s*\{[^}]*\}\s*$", "", m.group(2)), []
        else:
            buf.append(line)
    if head: out.append((head, "\n".join(buf)))
    return out


def split_rst(text):
    lines, out, head, buf, i = text.splitlines(), [], None, [], 0
    while i < len(lines):
        if i + 1 < len(lines) and lines[i].strip() and not RST_ADORN.match(lines[i]) and RST_ADORN.match(lines[i + 1]) \
                and len(lines[i + 1].strip()) >= len(lines[i].strip()) - 1 and not lines[i].startswith(" "):
            if head: out.append((head, "\n".join(buf)))
            head, buf, i = lines[i].strip(), [], i + 2
            continue
        buf.append(lines[i]); i += 1
    if head: out.append((head, "\n".join(buf)))
    return out


def jaccard(a, b):
    a, b = set(re.findall(r"[a-z0-9]{2,}", a.lower())), set(re.findall(r"[a-z0-9]{2,}", b.lower()))
    return len(a & b) / max(len(a | b), 1)


def main():
    secs = []
    for lib, vers in VERSIONS.items():
        for role, tag in zip(("old", "new"), vers):
            base = RAW / lib / tag
            for f in sorted(base.rglob("*")):
                if f.suffix not in (".md", ".rst") or SKIP_FILE.search(f.name):
                    continue
                txt = f.read_text(encoding="utf-8", errors="ignore")
                for head, body in (split_md(txt) if f.suffix == ".md" else split_rst(txt)):
                    body = clean(body); w = body.split()
                    if len(w) < MIN_WORDS: continue
                    secs.append(dict(lib=lib, role=role, version=tag, file=str(f.relative_to(base)),
                                     heading=clean(head), key=norm_head(head), text=" ".join(w[:MAX_WORDS])))
    for i, s in enumerate(secs): s["id"] = i
    # pair across versions
    pairs, stats = [], {}
    for lib in VERSIONS:
        cnt = {r: {} for r in ("old", "new")}
        for s in secs:
            if s["lib"] == lib: cnt[s["role"]].setdefault(s["key"], []).append(s)
        for k, o in cnt["old"].items():
            n = cnt["new"].get(k)
            if len(o) == 1 and n and len(n) == 1 and k not in GENERIC and len(k) >= 6:
                j = jaccard(o[0]["text"], n[0]["text"])
                cls = "unchanged" if j >= 0.95 else "changed" if j < 0.80 else "partial"
                pairs.append(dict(lib=lib, key=k, old_id=o[0]["id"], new_id=n[0]["id"], jaccard=round(j, 3), cls=cls))
    (ROOT / "data").mkdir(exist_ok=True)
    with open(ROOT / "data" / "sections.jsonl", "w", encoding="utf-8") as f:
        for s in secs: f.write(json.dumps(s) + "\n")
    json.dump(pairs, open(ROOT / "data" / "pairs.json", "w"), indent=0)
    import collections
    print("sections:", collections.Counter((s["lib"], s["role"]) for s in secs))
    print("pairs   :", collections.Counter((p["lib"], p["cls"]) for p in pairs))


if __name__ == "__main__":
    main()
