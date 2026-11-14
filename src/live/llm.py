"""LLM client with per-question metering (tokens, cost, latency).

Supports both Anthropic Claude and Google Gemini APIs.
Prices are NOT hard-coded: pass the current per-million-token prices for the model
you run (--price-in / --price-out). A dry run uses a deterministic stub and no network.
"""
import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional


def parse_json(text: str) -> dict:
    """Extract the first JSON object from a model reply; {} if none parses."""
    if not text:
        return {}
    a, b = text.find("{"), text.rfind("}")
    if a == -1 or b <= a:
        return {}
    try:
        obj = json.loads(text[a:b + 1])
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


@dataclass
class Meter:
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    llm_seconds: float = 0.0


@dataclass
class LLM:
    """Unified LLM client supporting Anthropic, Google Gemini, and Groq."""
    model: str
    price_in: float = 0.0   # USD per 1M input tokens
    price_out: float = 0.0  # USD per 1M output tokens
    temperature: float = 0.0
    max_retries: int = 5
    meter: Meter = field(default_factory=Meter)
    _client: object = None
    _provider: str = "anthropic"  # "anthropic", "gemini", or "groq"

    def __post_init__(self):
        # Auto-detect provider based on model name or env vars
        model_lower = self.model.lower()
        if "groq" in model_lower or os.environ.get("GROQ_API_KEY"):
            self._provider = "groq"
            self._init_groq()
        elif "gemini" in model_lower or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
            self._provider = "gemini"
            self._init_gemini()
        else:
            self._provider = "anthropic"
            self._init_anthropic()

    def _init_anthropic(self):
        """Initialize Anthropic client."""
        import anthropic
        self._client = anthropic.Anthropic(max_retries=self.max_retries)

    def _init_gemini(self):
        """Initialize Google Gemini client."""
        try:
            import google.generativeai as genai
            api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set")
            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(self.model)
        except ImportError:
            raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")

    def _init_groq(self):
        """Initialize Groq client (OpenAI-compatible API)."""
        try:
            from openai import OpenAI
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable not set")
            self._client = OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
            )
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

    def reset(self):
        self.meter = Meter()

    def cost(self) -> float:
        return (self.meter.tokens_in * self.price_in + self.meter.tokens_out * self.price_out) / 1e6

    def call(self, system: str, user: str, max_tokens: int = 1024) -> str:
        """Make an LLM call with the given system and user prompts."""
        t0 = time.perf_counter()
        
        if self._provider == "gemini":
            return self._call_gemini(system, user, max_tokens, t0)
        elif self._provider == "groq":
            return self._call_groq(system, user, max_tokens, t0)
        else:
            return self._call_anthropic(system, user, max_tokens, t0)

    def _call_anthropic(self, system: str, user: str, max_tokens: int, t0: float) -> str:
        """Call Anthropic Claude API."""
        resp = self._client.messages.create(
            model=self.model, max_tokens=max_tokens, temperature=self.temperature,
            system=system, messages=[{"role": "user", "content": user}],
        )
        self.meter.llm_seconds += time.perf_counter() - t0
        self.meter.calls += 1
        self.meter.tokens_in += resp.usage.input_tokens
        self.meter.tokens_out += resp.usage.output_tokens
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

    def _call_gemini(self, system: str, user: str, max_tokens: int, t0: float) -> str:
        """Call Google Gemini API with rate limiting for free tier."""
        # Combine system and user prompts for Gemini
        full_prompt = f"{system}\n\n{user}"
        
        # Configure generation with timeout
        generation_config = {
            "temperature": self.temperature,
            "max_output_tokens": min(max_tokens, 2048),  # Limit to avoid timeouts
        }
        
        # Make the call with retry logic and rate limiting
        import time
        max_retries = 5
        base_delay = 5  # Start with 5 second delay for rate limiting
        
        for attempt in range(max_retries):
            try:
                # Add delay before each call to respect rate limits
                if attempt > 0:
                    delay = base_delay * (2 ** (attempt - 1))  # Exponential backoff
                    print(f"Rate limit hit, waiting {delay}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(delay)
                
                response = self._client.generate_content(
                    full_prompt,
                    generation_config=generation_config,
                )
                break
            except Exception as e:
                error_str = str(e).lower()
                if "rate limit" in error_str or "quota" in error_str or "429" in error_str:
                    if attempt == max_retries - 1:
                        raise
                    # Extract retry delay from error if available
                    import re
                    retry_match = re.search(r'retry in ([\d.]+)s', str(e))
                    if retry_match:
                        delay = float(retry_match.group(1)) + 1  # Add 1s buffer
                        print(f"Rate limit hit, waiting {delay}s (from API)...")
                        time.sleep(delay)
                    else:
                        delay = base_delay * (2 ** attempt)
                        print(f"Rate limit hit, waiting {delay}s before retry {attempt + 1}/{max_retries}...")
                        time.sleep(delay)
                else:
                    if attempt == max_retries - 1:
                        raise
                    print(f"Gemini API error (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        self.meter.llm_seconds += time.perf_counter() - t0
        self.meter.calls += 1
        
        # Extract text and token counts
        text = response.text if response.text else ""
        
        # Try to get token counts from response metadata
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            self.meter.tokens_in += getattr(usage, 'prompt_token_count', 0)
            self.meter.tokens_out += getattr(usage, 'candidates_token_count', 0)
        else:
            # Fallback: estimate tokens (rough approximation)
            self.meter.tokens_in += len(full_prompt.split())
            self.meter.tokens_out += len(text.split())
        
        return text

    def _call_groq(self, system: str, user: str, max_tokens: int, t0: float) -> str:
        """Call Groq API (OpenAI-compatible)."""
        # Combine system and user prompts for Groq
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        
        # Make the call with retry logic
        import time
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=min(max_tokens, 2048),
                    temperature=self.temperature,
                )
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"Groq API error (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
        
        self.meter.llm_seconds += time.perf_counter() - t0
        self.meter.calls += 1
        
        # Extract text and token counts
        text = response.choices[0].message.content if response.choices else ""
        
        # Get token counts from response
        if hasattr(response, 'usage'):
            self.meter.tokens_in += response.usage.prompt_tokens
            self.meter.tokens_out += response.usage.completion_tokens
        else:
            # Fallback: estimate tokens
            self.meter.tokens_in += len(system.split()) + len(user.split())
            self.meter.tokens_out += len(text.split())
        
        return text


class StubLLM:
    """Deterministic offline stand-in used ONLY to test the harness plumbing.
    Its outputs carry no information about system quality; dry-run logs are
    tagged run_type=dry_run and rejected by analyze_results.py."""
    model = "stub"
    price_in = price_out = 0.0

    def __init__(self):
        self.meter = Meter()

    def reset(self):
        self.meter = Meter()

    def cost(self) -> float:
        return 0.0

    def call(self, system: str, user: str, max_tokens: int = 1024) -> str:
        self.meter.calls += 1
        self.meter.tokens_in += len(user.split())
        self.meter.tokens_out += 20
        s = system.lower()
        if "judge" in s:
            return '{"verdict": "incorrect", "reason": "stub"}'
        if "query analyst" in s:
            return '{"library": "unknown", "target_version": "latest", "sub_queries": ["stub query"]}'
        if "evidence analyst" in s:
            return '{"sufficiency": "sufficient", "relevant_docs": []}'
        if "conflict detector" in s:
            return '{"conflicts": []}'
        if "verifier" in s or "critic" in s:
            return '{"supported": true, "claims": [], "critique": "ok", "revised_answer": null}'
        return '{"answer": "stub answer", "citations": [], "abstain": false}'
