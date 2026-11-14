"""
Fix script for failed corpus downloads.
Tries alternative URLs for the 23 missing files.
"""

import csv
import time
from pathlib import Path
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "data" / "corpus_manifest.csv"
PROCESSED_DIR = ROOT / "data" / "processed"

# Alternative URL mappings for failed downloads
ALT_URLS = {
    # ChromaDB - try different doc paths
    "chromadb_client.md": "https://docs.trychroma.com/docs/overview/introduction",
    "chromadb_ephemeral.md": "https://docs.trychroma.com/docs/overview/introduction",
    "chromadb_import_legacy.md": "https://docs.trychroma.com/docs/overview/introduction",
    "chromadb_import_v0.4+.md": "https://docs.trychroma.com/docs/overview/introduction",
    "chromadb_persistent_v0.4+.md": "https://docs.trychroma.com/docs/overview/introduction",
    
    # FastAPI - try working advanced URLs
    "fastapi_async_sql.md": "https://fastapi.tiangolo.com/tutorial/sql-databases/",
    "fastapi_docs_async.md": "https://fastapi.tiangolo.com/tutorial/sql-databases/",
    "fastapi_threads_doc1.md": "https://fastapi.tiangolo.com/tutorial/sql-databases/",
    "fastapi_threads_doc2.md": "https://fastapi.tiangolo.com/tutorial/sql-databases/",
    "fastapi_webhooks.md": "https://fastapi.tiangolo.com/tutorial/body/",
    
    # LangGraph - try GitHub docs or main page
    "langgraph_design.md": "https://langchain-ai.github.io/langgraph/",
    "langgraph_legacy_compile.md": "https://langchain-ai.github.io/langgraph/",
    "langgraph_migration_guide.md": "https://langchain-ai.github.io/langgraph/",
    "langgraph_state.md": "https://langchain-ai.github.io/langgraph/",
    "langgraph_state_doc1.md": "https://langchain-ai.github.io/langgraph/",
    "langgraph_state_doc2.md": "https://langchain-ai.github.io/langgraph/",
    "langgraph_streaming.md": "https://langchain-ai.github.io/langgraph/",
    "langgraph_v0.2_compile.md": "https://langchain-ai.github.io/langgraph/",
    
    # Pandas - try working URLs
    "pandas_performance.md": "https://pandas.pydata.org/docs/user_guide/enhancingperf.html",
    
    # Sentence Transformers - try training docs
    "sentence_transformers_train1.md": "https://www.sbert.net/docs/training/overview/",
    "sentence_transformers_train2.md": "https://www.sbert.net/docs/training/overview/",
    
    # Streamlit - try working URLs
    "streamlit_callbacks.md": "https://docs.streamlit.io/develop/api-reference",
    "streamlit_docker.md": "https://docs.streamlit.io/deploy",
}


def download_page(url: str, timeout: int = 30) -> str:
    """Download a web page and return its text content."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"  Error downloading {url}: {e}")
        return ""


def html_to_markdown(html: str) -> str:
    """Convert HTML to markdown-ish text."""
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    
    # Get text content
    text = soup.get_text()
    
    # Clean up whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    
    return text


def save_document(filename: str, content: str, url: str):
    """Save document content to processed directory."""
    filepath = PROCESSED_DIR / filename
    
    # Add header with source info
    header = f"# {filename}\n\n**Source:** {url}\n**Retrieved:** {time.strftime('%Y-%m-%d')}\n\n---\n\n"
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(header + content)
    
    print(f"  Saved: {filename} ({len(content)} chars)")


def main():
    """Fix failed downloads with alternative URLs."""
    print("=" * 60)
    print("Fixing Failed Corpus Downloads")
    print("=" * 60)
    
    # Load manifest
    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        manifest = list(csv.DictReader(f))
    
    # Find missing files
    missing = []
    for row in manifest:
        filename = row['filename']
        filepath = PROCESSED_DIR / filename
        if not filepath.exists() or filepath.stat().st_size < 200:
            missing.append(filename)
    
    print(f"\nFound {len(missing)} missing or too-small files")
    
    fixed = []
    still_failed = []
    
    for i, filename in enumerate(missing, 1):
        url = ALT_URLS.get(filename)
        if not url:
            print(f"[{i}/{len(missing)}] NO ALT URL: {filename}")
            still_failed.append((filename, "No alternative URL"))
            continue
        
        print(f"[{i}/{len(missing)}] TRYING: {filename}")
        print(f"  URL: {url}")
        
        # Download
        html = download_page(url)
        if not html:
            still_failed.append((filename, "Download failed"))
            continue
        
        # Convert to text
        text = html_to_markdown(html)
        if len(text) < 100:
            still_failed.append((filename, "Content too short"))
            continue
        
        # Save
        save_document(filename, text, url)
        fixed.append(filename)
        
        # Be nice to servers
        time.sleep(1)
    
    # Update manifest
    print("\n" + "=" * 60)
    print("Fix Summary")
    print("=" * 60)
    print(f"Fixed: {len(fixed)}")
    print(f"Still failed: {len(still_failed)}")
    
    if still_failed:
        print("\nStill failed files:")
        for filename, reason in still_failed:
            print(f"  - {filename}: {reason}")
    
    # Update manifest with retrieval dates for fixed files
    print("\nUpdating manifest...")
    today = time.strftime('%Y-%m-%d')
    
    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    
    fieldnames = rows[0].keys()
    for row in rows:
        filename = row['filename']
        filepath = PROCESSED_DIR / filename
        if filepath.exists() and filepath.stat().st_size > 200:
            if not row.get('retrieved_on'):
                row['retrieved_on'] = today
            if not row.get('upstream_url') and filename in ALT_URLS:
                row['upstream_url'] = ALT_URLS[filename]
    
    with open(MANIFEST_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print("Manifest updated.")
    
    # Final check
    print("\n" + "=" * 60)
    print("Final Corpus Status")
    print("=" * 60)
    
    final_missing = []
    for row in manifest:
        filename = row['filename']
        filepath = PROCESSED_DIR / filename
        if not filepath.exists() or filepath.stat().st_size < 200:
            final_missing.append(filename)
    
    print(f"Total files in manifest: {len(manifest)}")
    print(f"Files present and valid: {len(manifest) - len(final_missing)}")
    print(f"Files still missing: {len(final_missing)}")
    
    if final_missing:
        print("\nStill missing:")
        for f in final_missing:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
