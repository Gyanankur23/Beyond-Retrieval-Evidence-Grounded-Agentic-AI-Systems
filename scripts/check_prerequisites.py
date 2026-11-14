"""
Live evaluation runner helper script.
Checks prerequisites and provides clear instructions for running the evaluation.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def check_prerequisites():
    """Check if all prerequisites are met for live evaluation."""
    print("=" * 60)
    print("Live Evaluation Prerequisites Check")
    print("=" * 60)
    
    issues = []
    
    # Check API keys (Anthropic, Gemini, or Groq)
    anthropic_key = os.environ.get('ANTHROPIC_API_KEY')
    gemini_key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
    groq_key = os.environ.get('GROQ_API_KEY')
    
    if not anthropic_key and not gemini_key and not groq_key:
        issues.append("No LLM API key set (need ANTHROPIC_API_KEY, GOOGLE_API_KEY/GEMINI_API_KEY, or GROQ_API_KEY)")
        print("[FAIL] No LLM API key set")
        print("       Set ANTHROPIC_API_KEY for Claude, GOOGLE_API_KEY/GEMINI_API_KEY for Gemini, or GROQ_API_KEY for Groq")
    elif anthropic_key:
        print("[PASS] ANTHROPIC_API_KEY is set (Claude available)")
    elif gemini_key:
        print("[PASS] GOOGLE_API_KEY/GEMINI_API_KEY is set (Gemini available)")
    elif groq_key:
        print("[PASS] GROQ_API_KEY is set (Groq available)")
    
    # Check corpus
    processed_dir = ROOT / "data" / "processed"
    manifest_path = ROOT / "data" / "corpus_manifest.csv"
    
    if not manifest_path.exists():
        issues.append("corpus_manifest.csv not found")
        print("[FAIL] corpus_manifest.csv not found")
    else:
        import csv
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = list(csv.DictReader(f))
        
        missing = []
        unconfirmed = []
        for row in manifest:
            filename = row['filename']
            filepath = processed_dir / filename
            if not filepath.exists() or filepath.stat().st_size < 200:
                missing.append(filename)
            elif row.get('confirmed', '').lower() != 'yes':
                unconfirmed.append(filename)
        
        if missing:
            issues.append(f"{len(missing)} corpus files missing or too small")
            print(f"[FAIL] {len(missing)} corpus files missing or too small")
        elif unconfirmed:
            issues.append(f"{len(unconfirmed)} corpus files not confirmed")
            print(f"[FAIL] {len(unconfirmed)} corpus files not confirmed in manifest")
        else:
            print(f"[PASS] All {len(manifest)} corpus files present and confirmed")
    
    # Check questions file
    questions_path = ROOT / "data" / "questions.json"
    if not questions_path.exists():
        issues.append("questions.json not found")
        print("[FAIL] questions.json not found")
    else:
        import json
        with open(questions_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        print(f"[PASS] questions.json loaded ({len(questions)} questions)")
    
    # Check dependencies
    try:
        import anthropic
        import chromadb
        print("[PASS] Core dependencies installed (anthropic, chromadb)")
    except ImportError as e:
        issues.append(f"Missing core dependency: {e}")
        print(f"[FAIL] Missing core dependency: {e}")
    
    # Check google-generativeai (for Gemini)
    try:
        import google.generativeai
        print("[PASS] google-generativeai installed (Gemini available)")
    except ImportError:
        print("[WARN] google-generativeai not installed")
        print("       Install with: pip install google-generativeai")
    
    # Check openai (for Groq)
    try:
        import openai
        print("[PASS] openai installed (Groq available)")
    except ImportError:
        print("[WARN] openai not installed")
        print("       Install with: pip install openai")
    
    # Check sentence_transformers (optional - can use TF-IDF backend)
    try:
        import sentence_transformers
        print("[PASS] sentence_transformers installed (MiniLM backend available)")
    except (ImportError, OSError) as e:
        print(f"[WARN] sentence_transformers not available: {e}")
        print("       Will use TF-IDF backend instead (slightly lower quality but works)")
    
    print("\n" + "=" * 60)
    if issues:
        print("PREREQUISITES NOT MET")
        print("=" * 60)
        for issue in issues:
            print(f"  - {issue}")
        print("\nPlease fix the issues above before running live evaluation.")
        return False
    else:
        print("ALL PREREQUISITES MET")
        print("=" * 60)
        print("\nYou can now run the live evaluation.")
        print("\nOption 1: Using Groq (FREE tier - RECOMMENDED, fast & no rate limits):")
        print("  python experiments/run_evaluations.py \\")
        print("      --config all \\")
        print("      --backend tfidf \\")
        print("      --model llama-3.3-70b-versatile \\")
        print("      --judge-model llama-3.3-70b-versatile \\")
        print("      --price-in 0.00 \\")
        print("      --price-out 0.00")
        print("\nOption 2: Using Google Gemini (FREE tier, but rate limited):")
        print("  python experiments/run_evaluations.py \\")
        print("      --config all \\")
        print("      --backend tfidf \\")
        print("      --model gemini-flash-lite-latest \\")
        print("      --judge-model gemini-flash-lite-latest \\")
        print("      --price-in 0.00 \\")
        print("      --price-out 0.00 \\")
        print("      --delay 5")
        print("\nOption 3: Using Anthropic Claude (paid):")
        print("  python experiments/run_evaluations.py \\")
        print("      --config all \\")
        print("      --backend tfidf \\")
        print("      --model claude-sonnet-4-20250514 \\")
        print("      --judge-model claude-3-5-haiku-20241022 \\")
        print("      --price-in 3.00 \\")
        print("      --price-out 15.00")
        print("\nOr run only main configurations first (Groq):")
        print("  python experiments/run_evaluations.py \\")
        print("      --config main \\")
        print("      --backend tfidf \\")
        print("      --model llama-3.3-70b-versatile \\")
        print("      --judge-model llama-3.3-70b-versatile \\")
        print("      --price-in 0.00 \\")
        print("      --price-out 0.00")
        return True


def main():
    """Main function."""
    success = check_prerequisites()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
