"""
Automated corpus downloader for the live evaluation harness.

This script attempts to download real documentation from official sources
for the 157 files required by the benchmark. It maps filenames to URLs
and saves content in markdown format.

NOTE: This is a best-effort automated approach. Some files may need
manual verification or alternative sources.
"""

import csv
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "data" / "corpus_manifest.csv"
PROCESSED_DIR = ROOT / "data" / "processed"

# Create processed directory if it doesn't exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# URL mapping for documentation sources
# This is a best-effort mapping - some URLs may need adjustment
URL_MAPPINGS = {
    # FastAPI
    "fastapi_routing.md": "https://fastapi.tiangolo.com/tutorial/dependencies/",
    "fastapi_cors.md": "https://fastapi.tiangolo.com/tutorial/cors/",
    "fastapi_exceptions.md": "https://fastapi.tiangolo.com/tutorial/handling-errors/",
    "fastapi_responses.md": "https://fastapi.tiangolo.com/tutorial/response-model/",
    "fastapi_security.md": "https://fastapi.tiangolo.com/tutorial/security/",
    "fastapi_testing.md": "https://fastapi.tiangolo.com/tutorial/testing/",
    "fastapi_deployment.md": "https://fastapi.tiangolo.com/deployment/",
    "fastapi_docker.md": "https://fastapi.tiangolo.com/deployment/docker/",
    "fastapi_async_sql.md": "https://fastapi.tiangolo.com/advanced/async-sql-databases/",
    "fastapi_validation.md": "https://fastapi.tiangolo.com/tutorial/body/",
    "fastapi_serialization.md": "https://fastapi.tiangolo.com/tutorial/response-model/",
    "fastapi_webhooks.md": "https://fastapi.tiangolo.com/advanced/webhooks/",
    "fastapi_lifespan_modern.md": "https://fastapi.tiangolo.com/advanced/events/",
    "fastapi_lifespan_v0.100+.md": "https://fastapi.tiangolo.com/advanced/events/",
    "fastapi_events_legacy.md": "https://fastapi.tiangolo.com/advanced/events/",
    "fastapi_events_v0.90.md": "https://fastapi.tiangolo.com/advanced/events/",
    "fastapi_oauth2_legacy.md": "https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/",
    "fastapi_oauth2_modern.md": "https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/",
    "fastapi_response_model_old.md": "https://fastapi.tiangolo.com/tutorial/response-model/",
    "fastapi_response_model_v0.100+.md": "https://fastapi.tiangolo.com/tutorial/response-model/",
    "fastapi_alias1.md": "https://fastapi.tiangolo.com/tutorial/path-params/",
    "fastapi_alias2.md": "https://fastapi.tiangolo.com/tutorial/path-params/",
    "fastapi_docs_async.md": "https://fastapi.tiangolo.com/advanced/async-sql-databases/",
    "fastapi_docs_sync.md": "https://fastapi.tiangolo.com/tutorial/sql-databases/",
    "fastapi_security_basic.md": "https://fastapi.tiangolo.com/tutorial/security/simple-oauth2/",
    "fastapi_testclient.md": "https://fastapi.tiangolo.com/tutorial/testing/",
    "fastapi_testclient_doc1.md": "https://fastapi.tiangolo.com/tutorial/testing/",
    "fastapi_testclient_doc2.md": "https://fastapi.tiangolo.com/tutorial/testing/",
    "fastapi_threads_doc1.md": "https://fastapi.tiangolo.com/advanced/async-sql-databases/",
    "fastapi_threads_doc2.md": "https://fastapi.tiangolo.com/advanced/async-sql-databases/",
    
    # LangChain
    "langchain_overview.md": "https://python.langchain.com/docs/get_started/introduction/",
    "langchain_chat_models.md": "https://python.langchain.com/docs/modules/model_io/chat/",
    "langchain_retrievers.md": "https://python.langchain.com/docs/modules/data_connection/retrievers/",
    "langchain_prompts.md": "https://python.langchain.com/docs/modules/model_io/prompts/",
    "langchain_text_splitters.md": "https://python.langchain.com/docs/modules/data_connection/document_transformers/",
    "langchain_lcel.md": "https://python.langchain.com/docs/expression_language/",
    "langchain_lcel_guide.md": "https://python.langchain.com/docs/expression_language/",
    "langchain_lcel_async.md": "https://python.langchain.com/docs/expression_language/async/",
    "langchain_lcel_sync.md": "https://python.langchain.com/docs/expression_language/sync/",
    "langchain_callbacks.md": "https://python.langchain.com/docs/modules/callbacks/",
    "langchain_logging.md": "https://python.langchain.com/docs/modules/callbacks/",
    "langchain_config.md": "https://python.langchain.com/docs/get_started/configuration/",
    "langchain_caching.md": "https://python.langchain.com/docs/modules/caching/",
    "langchain_agent_executor.md": "https://python.langchain.com/docs/modules/agents/agent_types/",
    "langchain_maintainers.md": "https://python.langchain.com/docs/contributing/",
    "langchain_prompt_doc1.md": "https://python.langchain.com/docs/modules/model_io/prompts/",
    "langchain_prompt_doc2.md": "https://python.langchain.com/docs/modules/model_io/prompts/",
    "langchain_chain_legacy.md": "https://python.langchain.com/docs/modules/chains/",
    "langchain_legacy_chain.md": "https://python.langchain.com/docs/modules/chains/",
    "langchain_stop_legacy.md": "https://python.langchain.com/docs/modules/agents/agent_types/",
    "langchain_stop_v0.2.md": "https://python.langchain.com/docs/modules/agents/agent_types/",
    "langchain_tools_v0.1.md": "https://python.langchain.com/docs/modules/agents/tools/",
    "langchain_tools_v0.2.md": "https://python.langchain.com/docs/modules/agents/tools/",
    
    # LangGraph
    "langgraph_core.md": "https://langchain-ai.github.io/langgraph/",
    "langgraph_design.md": "https://langchain-ai.github.io/langgraph/concepts/low_level/",
    "langgraph_state.md": "https://langchain-ai.github.io/langgraph/concepts/low_level/",
    "langgraph_state_doc1.md": "https://langchain-ai.github.io/langgraph/concepts/low_level/",
    "langgraph_state_doc2.md": "https://langchain-ai.github.io/langgraph/concepts/low_level/",
    "langgraph_streaming.md": "https://langchain-ai.github.io/langgraph/concepts/low_level/",
    "langgraph_migration_guide.md": "https://langchain-ai.github.io/langgraph/concepts/migration/",
    "langgraph_legacy_compile.md": "https://langchain-ai.github.io/langgraph/concepts/low_level/",
    "langgraph_v0.2_compile.md": "https://langchain-ai.github.io/langgraph/concepts/low_level/",
    
    # ChromaDB
    "chromadb_about.md": "https://docs.trychroma.com/docs/overview/introduction",
    "chromadb_architecture.md": "https://docs.trychroma.com/docs/overview/architecture",
    "chromadb_client.md": "https://docs.trychroma.com/docs/getting-started",
    "chromadb_user_guide.md": "https://docs.trychroma.com/docs/overview/introduction",
    "chromadb_queries.md": "https://docs.trychroma.com/docs/querying-collections/query-and-get",
    "chromadb_batch.md": "https://docs.trychroma.com/docs/collections/add-data",
    "chromadb_delete_legacy.md": "https://docs.trychroma.com/docs/collections/delete-data",
    "chromadb_delete_v0.4+.md": "https://docs.trychroma.com/docs/collections/delete-data",
    "chromadb_dims_legacy.md": "https://docs.trychroma.com/docs/collections/configure",
    "chromadb_dims_v0.4+.md": "https://docs.trychroma.com/docs/collections/configure",
    "chromadb_import_legacy.md": "https://docs.trychroma.com/docs/collections/import-export",
    "chromadb_import_v0.4+.md": "https://docs.trychroma.com/docs/collections/import-export",
    "chromadb_persistent_v0.4+.md": "https://docs.trychroma.com/docs/deployment/deploy",
    "chromadb_ephemeral.md": "https://docs.trychroma.com/docs/deployment/deploy",
    "chromadb_testing.md": "https://docs.trychroma.com/docs/overview/introduction",
    "chromadb_threads_async.md": "https://docs.trychroma.com/docs/overview/introduction",
    "chromadb_threads_sync.md": "https://docs.trychroma.com/docs/overview/introduction",
    "chromadb_storage_doc1.md": "https://docs.trychroma.com/docs/overview/architecture",
    "chromadb_storage_doc2.md": "https://docs.trychroma.com/docs/overview/architecture",
    "chromadb_voice.md": "https://docs.trychroma.com/docs/overview/introduction",
    
    # Pandas
    "pandas_groupby.md": "https://pandas.pydata.org/docs/user_guide/groupby.html",
    "pandas_groupby_legacy.md": "https://pandas.pydata.org/docs/user_guide/groupby.html",
    "pandas_groupby_v2.2.md": "https://pandas.pydata.org/docs/user_guide/groupby.html",
    "pandas_merge.md": "https://pandas.pydata.org/docs/user_guide/merging.html",
    "pandas_merge_doc1.md": "https://pandas.pydata.org/docs/user_guide/merging.html",
    "pandas_merge_doc2.md": "https://pandas.pydata.org/docs/user_guide/merging.html",
    "pandas_indexing.md": "https://pandas.pydata.org/docs/user_guide/indexing.html",
    "pandas_loc_v2.md": "https://pandas.pydata.org/docs/user_guide/indexing.html",
    "pandas_dtypes.md": "https://pandas.pydata.org/docs/user_guide/basics.html",
    "pandas_timeseries.md": "https://pandas.pydata.org/docs/user_guide/timeseries.html",
    "pandas_io.md": "https://pandas.pydata.org/docs/user_guide/io.html",
    "pandas_io_v1.md": "https://pandas.pydata.org/docs/user_guide/io.html",
    "pandas_io_v2.md": "https://pandas.pydata.org/docs/user_guide/io.html",
    "pandas_io_json.md": "https://pandas.pydata.org/docs/user_guide/io.html",
    "pandas_csv_infer1.md": "https://pandas.pydata.org/docs/user_guide/io.html",
    "pandas_csv_infer2.md": "https://pandas.pydata.org/docs/user_guide/io.html",
    "pandas_parquet.md": "https://pandas.pydata.org/docs/user_guide/io.html",
    "pandas_excel_legacy.md": "https://pandas.pydata.org/docs/user_guide/io.html",
    "pandas_excel_v2.md": "https://pandas.pydata.org/docs/user_guide/io.html",
    "pandas_export.md": "https://pandas.pydata.org/docs/user_guide/io.html",
    "pandas_chunking.md": "https://pandas.pydata.org/docs/user_guide/io.html",
    "pandas_iter_chunks.md": "https://pandas.pydata.org/docs/user_guide/io.html",
    "pandas_iterrows.md": "https://pandas.pydata.org/docs/user_guide/basics.html",
    "pandas_json_normalize.md": "https://pandas.pydata.org/docs/user_guide/io.html",
    "pandas_performance.md": "https://pandas.pydata.org/docs/user_guide/enhancingperf.html",
    "pandas_append_v1.md": "https://pandas.pydata.org/docs/user_guide/basics.html",
    "pandas_concat_v2.md": "https://pandas.pydata.org/docs/user_guide/basics.html",
    "pandas_copy_v1.md": "https://pandas.pydata.org/docs/user_guide/basics.html",
    "pandas_fillna_legacy.md": "https://pandas.pydata.org/docs/user_guide/missing_data.html",
    "pandas_ffill_modern.md": "https://pandas.pydata.org/docs/user_guide/missing_data.html",
    
    # Pydantic
    "pydantic_v2.md": "https://docs.pydantic.dev/latest/",
    "pydantic_v2_model_config.md": "https://docs.pydantic.dev/latest/concepts/models/",
    "pydantic_v1_config.md": "https://docs.pydantic.dev/latest/migration/",
    "pydantic_validation.md": "https://docs.pydantic.dev/latest/concepts/validators/",
    "pydantic_error_handling.md": "https://docs.pydantic.dev/latest/concepts/errors/",
    "pydantic_mutable_doc1.md": "https://docs.pydantic.dev/latest/concepts/models/",
    "pydantic_mutable_doc2.md": "https://docs.pydantic.dev/latest/concepts/models/",
    
    # Streamlit
    "streamlit_data.md": "https://docs.streamlit.io/develop/api-reference/data",
    "streamlit_widgets.md": "https://docs.streamlit.io/develop/api-reference/widgets",
    "streamlit_callbacks.md": "https://docs.streamlit.io/develop/api-reference/callbacks",
    "streamlit_docker.md": "https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/docker",
    "streamlit_exec_doc1.md": "https://docs.streamlit.io/develop/concepts/architecture/architecture",
    "streamlit_exec_doc2.md": "https://docs.streamlit.io/develop/concepts/architecture/architecture",
    
    # GitHub
    "github_rest_api.md": "https://docs.github.com/en/rest",
    "github_api_headers.md": "https://docs.github.com/en/rest/overview/resources-in-the-rest-api",
    "github_actions_triggers.md": "https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows",
    "github_actions_classic.md": "https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions",
    "github_actions_fine_grained.md": "https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions",
    "github_actions_infrastructure.md": "https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners",
    "github_enterprise_runners.md": "https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/about-self-hosted-runners",
    "github_infrastructure.md": "https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/about-self-hosted-runners",
    "github_secrets_doc1.md": "https://docs.github.com/en/actions/security-guides/encrypted-secrets",
    "github_secrets_doc2.md": "https://docs.github.com/en/actions/security-guides/encrypted-secrets",
    "github_sustainability.md": "https://docs.github.com/en/actions/learn-github-actions/variables",
    
    # GitHub CLI
    "gh_cli_auth1.md": "https://cli.github.com/manual/gh_auth_login",
    "gh_cli_auth2.md": "https://cli.github.com/manual/gh_auth_login",
    "gh_cli_reference.md": "https://cli.github.com/manual/",
    
    # HuggingFace
    "huggingface_hub.md": "https://huggingface.co/docs/hub/index",
    "huggingface_inference.md": "https://huggingface.co/docs/inference-providers/index",
    
    # Sentence Transformers
    "sentence_transformers.md": "https://www.sbert.net/docs/",
    "sentence_transformers_overview.md": "https://www.sbert.net/docs/",
    "sentence_transformers_encode.md": "https://www.sbert.net/docs/package_reference/sentence_transformer/",
    "sentence_transformers_batch.md": "https://www.sbert.net/docs/package_reference/sentence_transformer/",
    "sentence_transformers_init.md": "https://www.sbert.net/docs/package_reference/sentence_transformer/",
    "sentence_transformers_cache1.md": "https://www.sbert.net/docs/package_reference/sentence_transformer/",
    "sentence_transformers_cache2.md": "https://www.sbert.net/docs/package_reference/sentence_transformer/",
    "sentence_transformers_train1.md": "https://www.sbert.net/docs/training/overview/",
    "sentence_transformers_train2.md": "https://www.sbert.net/docs/training/overview/",
    "sentence_transformers_security.md": "https://www.sbert.net/docs/package_reference/sentence_transformer/",
    "sentence_transformers_legacy.md": "https://www.sbert.net/docs/package_reference/sentence_transformer/",
    "sentence_transformers_v3.md": "https://www.sbert.net/docs/",
    
    # SQLite
    "sqlite_pandas.md": "https://pandas.pydata.org/docs/user_guide/io.html#sql-queries",
    
    # SQLAlchemy
    "sqlalchemy_asyncio.md": "https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html",
    
    # Next.js
    "nextjs_api_routes.md": "https://nextjs.org/docs/app/building-your-application/routing/route-handlers",
    
    # Python
    "python_history.md": "https://docs.python.org/3/whatsnew/",
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
    """Main function to download all documents."""
    print("=" * 60)
    print("Automated Corpus Downloader")
    print("=" * 60)
    
    # Load manifest
    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        manifest = list(csv.DictReader(f))
    
    print(f"\nFound {len(manifest)} files in manifest")
    print(f"URL mappings available: {len(URL_MAPPINGS)}")
    
    # Track results
    downloaded = []
    failed = []
    skipped = []
    
    for i, row in enumerate(manifest, 1):
        filename = row['filename']
        filepath = PROCESSED_DIR / filename
        
        # Skip if already exists and is substantial
        if filepath.exists() and filepath.stat().st_size > 200:
            print(f"[{i}/{len(manifest)}] SKIP: {filename} (already exists)")
            skipped.append(filename)
            continue
        
        # Get URL
        url = URL_MAPPINGS.get(filename)
        if not url:
            print(f"[{i}/{len(manifest)}] NO URL: {filename}")
            failed.append((filename, "No URL mapping"))
            continue
        
        print(f"[{i}/{len(manifest)}] DOWNLOADING: {filename}")
        print(f"  URL: {url}")
        
        # Download
        html = download_page(url)
        if not html:
            failed.append((filename, "Download failed"))
            continue
        
        # Convert to text
        text = html_to_markdown(html)
        if len(text) < 100:
            failed.append((filename, "Content too short"))
            continue
        
        # Save
        save_document(filename, text, url)
        downloaded.append(filename)
        
        # Be nice to servers
        time.sleep(1)
    
    # Update manifest with download info
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"Downloaded: {len(downloaded)}")
    print(f"Skipped (already exist): {len(skipped)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print("\nFailed files:")
        for filename, reason in failed[:20]:
            print(f"  - {filename}: {reason}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")
    
    # Update manifest with retrieved_on for downloaded files
    print("\nUpdating manifest with retrieval dates...")
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
            if not row.get('upstream_url') and filename in URL_MAPPINGS:
                row['upstream_url'] = URL_MAPPINGS[filename]
    
    with open(MANIFEST_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print("Manifest updated.")
    print("\nNext steps:")
    print("1. Review downloaded files for quality")
    print("2. Manually verify version-specific files (legacy vs current)")
    print("3. Set confirmed=yes in manifest for verified files")
    print("4. Run: python scripts/make_manifest.py to regenerate if needed")


if __name__ == "__main__":
    main()
