"""Create data/corpus_manifest.csv with one row per source file the benchmark references.

doc_version / status are PROPOSED from the filename only. The authors must verify each row
against the actual upstream document and set confirmed=yes; the runner refuses unconfirmed rows.
upstream_url and retrieved_on must record where each snapshot in data/processed/ came from.
"""
import csv, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
q = json.load(open(ROOT / "data" / "questions.json", encoding="utf-8"))
names = sorted({s for x in q for s in x["target_sources"]})
DEPRECATED = re.compile(r"(legacy|_old|v0\.90|_v1\b|_v0\.1\b)")
CURRENT = re.compile(r"(modern|v0\.100\+|_v2|v0\.2|v0\.4\+|_v3|v2\.2|v0\.2_)")
rows = []
for n in names:
    lib = n.split("_")[0]
    ver = re.search(r"(v\d[\d.]*\+?)", n)
    status = "deprecated" if DEPRECATED.search(n) else "current" if CURRENT.search(n) else "unspecified"
    rows.append(dict(filename=n, library=lib, doc_version=ver.group(1) if ver else "latest",
                     status=status, upstream_url="", retrieved_on="", confirmed=""))
with open(ROOT / "data" / "corpus_manifest.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} rows written; status counts:",
      {s: sum(r['status'] == s for r in rows) for s in ('current', 'deprecated', 'unspecified')})
