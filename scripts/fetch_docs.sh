#!/bin/bash
# Re-download the documentation snapshots at the pinned release tags (shallow + sparse git clones).
# Not needed to reproduce results: the Markdown/reStructuredText snapshots are already in data/raw_docs/.
cd "$(dirname "$0")/.."
fetch () {  # lib repo tag path
  out=data/raw_docs/$1/$3; rm -rf "$out"; mkdir -p "$out"; tmp=$(mktemp -d)
  git clone -q --depth 1 --filter=blob:none --sparse --branch "$3" "https://github.com/$2.git" "$tmp/r" &&
  (cd "$tmp/r" && git sparse-checkout set "$4") && cp -r "$tmp/r/$4/." "$out/"
  find "$out" -type f ! \( -name '*.md' -o -name '*.rst' \) -delete
  echo "$1 $3: $(find "$out" -type f | wc -l) files"; rm -rf "$tmp"
}
fetch fastapi  fastapi/fastapi             0.70.0   docs/en/docs
fetch fastapi  fastapi/fastapi             0.115.0  docs/en/docs
fetch pandas   pandas-dev/pandas           v1.5.3   doc/source/user_guide
fetch pandas   pandas-dev/pandas           v2.2.0   doc/source/user_guide
fetch sklearn  scikit-learn/scikit-learn   0.24.2   doc/modules
fetch sklearn  scikit-learn/scikit-learn   1.4.0    doc/modules
fetch pydantic pydantic/pydantic           v1.10.13 docs
fetch pydantic pydantic/pydantic           v2.5.0   docs
