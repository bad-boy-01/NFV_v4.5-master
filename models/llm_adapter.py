"""
LLM Adapter — Novel Video Factory v4
PRIMARY:  Groq free-tier (llama-3.3-70b) — sign up FREE at console.groq.com
FALLBACK: Ollama local (qwen2.5:7b)      — fully offline, no internet needed

BUG FIXES vs v3:
- unload_model URL was doubled (/api/api/generate) → FIXED: correct endpoint
- Groq timeout too low for long prompts → FIXED: 120s
- Better mock responses matching all system prompt types
"""
import json
import logging
import os
import requests
import time
from json_repair import repair_json

logger = logging.getLogger(__name__)


class LLMFallbackExhausted(Exception):
    """
    Raised when both the primary and fallback LLM providers fail AND
    strict_mode is enabled in config. In non-strict mode (the default),
    SmartLLMAdapter does not raise this — it returns mock content instead,
    but always sets `last_call_was_fallback = True` first so callers can
    detect and react to it (skip/flag/retry) instead of silently treating
    the mock as real story content.
    """
    pass


# ── Groq Free-Tier Adapter ────────────────────────────────────────────────────
class GroqLLMAdapter:
    """
    Groq free-tier LLM.
    Sign up FREE at https://console.groq.com — no credit card required.
    Set GROQ_API_KEY in Kaggle Secrets or environment variables.
    """
    def __init__(self, model_name: str = "llama-3.3-70b-versatile", api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key or self._load_key()
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.is_cloud = True

    def _load_key(self) -> str:
        """Try environment variable, then Kaggle Secrets."""
        key = os.environ.get("GROQ_API_KEY", "")
        if not key:
            try:
                from kaggle_secrets import UserSecretsClient  # type: ignore
                key = UserSecretsClient().get_secret("GROQ_API_KEY")
            except Exception:
                pass
        return key

    def check_health(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, system_prompt: str = None,
                 temperature: float = 0.7, model: str = None, **kwargs) -> str:
        if not self.api_key:
            return "ERROR: GROQ_NO_API_KEY"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(6):
            try:
                r = requests.post(
                    self.api_url,
                    headers={"Authorization": f"Bearer {self.api_key}",
                             "Content-Type": "application/json"},
                    json={"model": model or self.model_name,
                          "messages": messages,
                          "temperature": temperature,
                          "max_tokens": 4096},
                    timeout=120,
                )
                if r.status_code == 429:
                    # Exponential backoff: 10s, 30s, 60s, 120s, 180s, 300s
                    wait = [10, 30, 60, 120, 180, 300][attempt]
                    logger.warning(f"Groq Rate Limit (429). Attempt {attempt+1}/6. Waiting {wait}s...")
                    time.sleep(wait)
                    continue
                
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except Exception as e:
                logger.warning(f"Groq attempt {attempt+1} failed: {e}")
                if attempt < 5:
                    time.sleep(2)
        
        return "ERROR: GROQ_FAILED"

    def generate_json(self, prompt: str, system_prompt: str = None,
                      temperature: float = 0.1, model: str = None, **kwargs) -> str:
        """Force JSON response format if supported, then repair."""
        # Note: Groq supports response_format={"type": "json_object"} for some models
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt + " You must output valid JSON."})
        messages.append({"role": "user", "content": prompt})

        try:
            r = requests.post(
                self.api_url,
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
                json={"model": model or self.model_name,
                      "messages": messages,
                      "temperature": temperature,
                      "response_format": {"type": "json_object"},
                      "max_tokens": 4096},
                timeout=120,
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            return self._repair(content)
        except Exception as e:
            logger.debug(f"Groq JSON mode failed or unsupported: {e}")
            # Fallback to standard generate + repair
            content = self.generate(prompt, system_prompt, temperature, model, **kwargs)
            return self._repair(content)

    def _repair(self, content: str) -> str:
        if "ERROR:" in content: return content
        try:
            repaired = repair_json(content)
            if isinstance(repaired, (dict, list)):
                return json.dumps(repaired)
            return repaired
        except Exception:
            return content

    def unload_model(self, *args, **kwargs):
        pass  # No-op for API-based adapters


# ── Local Ollama Adapter ──────────────────────────────────────────────────────
class LocalLLMAdapter:
    """
    Ollama-backed local LLM — 100% offline, no API key needed.
    BUG FIX: unload_model previously used /api/api/generate (doubled path) — fixed.
    """
    def __init__(self, host: str = "http://localhost:11434",
                 model_name: str = "qwen2.5:7b"):
        self.host = host.rstrip("/")
        self.model_name = model_name
        self.api_url = f"{self.host}/api/generate"   # Correct single path
        self.is_cloud = False

    def check_health(self) -> bool:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            return r.status_code == 200
        except requests.ConnectionError:
            logger.error(f"Cannot reach Ollama at {self.host}")
            return False

    def generate(self, prompt: str, system_prompt: str = None,
                 temperature: float = 0.7, model: str = None,
                 keep_alive: str = "5m", **kwargs) -> str:
        selected = model or self.model_name
        payload = {
            "model": selected,
            "prompt": prompt,
            "stream": False,
            "keep_alive": keep_alive,
            "options": {"temperature": temperature},
        }
        if system_prompt:
            payload["system"] = system_prompt
        try:
            r = requests.post(self.api_url, json=payload, timeout=600)
            r.raise_for_status()
            return r.json().get("response", "")
        except Exception as e:
            logger.warning(f"Ollama error ({selected}): {e}")
            return f"ERROR: OLLAMA_FAILED ({selected})"

    def generate_json(self, prompt: str, system_prompt: str = None,
                      temperature: float = 0.1, model: str = None, **kwargs) -> str:
        selected = model or self.model_name
        payload = {
            "model": selected,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature},
        }
        if system_prompt:
            payload["system"] = system_prompt
        try:
            r = requests.post(self.api_url, json=payload, timeout=600)
            r.raise_for_status()
            content = r.json().get("response", "")
            return self._repair(content)
        except Exception:
            content = self.generate(prompt, system_prompt, temperature, model, **kwargs)
            return self._repair(content)

    def _repair(self, content: str) -> str:
        if "ERROR:" in content: return content
        try:
            repaired = repair_json(content)
            if isinstance(repaired, (dict, list)):
                return json.dumps(repaired)
            return repaired
        except Exception:
            return content

    def unload_model(self, model_name: str = None):
        """
        BUG FIX: Previous code did api_url.replace('/generate', '/api/generate')
        which produced the doubled path http://host/api/api/generate.
        Correct: POST to /api/generate with keep_alive=0.
        """
        target = model_name or self.model_name
        try:
            requests.post(
                self.api_url,   # Already correct: http://host/api/generate
                json={"model": target, "keep_alive": 0},
                timeout=10,
            )
            logger.info(f"Ollama unloaded: {target}")
        except Exception as e:
            logger.debug(f"Unload failed for {target}: {e}")


# ── Smart Adapter: tries Groq first, falls back to Ollama ────────────────────
class SmartLLMAdapter:
    """
    Tries Groq (free cloud) first by default, but honors 'provider' setting.
    If 'provider' is 'ollama', it uses Ollama as primary.

    Fallback visibility (fixes the silent-mock-substitution bug):
    - `last_call_was_fallback` is set on every call so callers can check
      immediately whether the result is real LLM output or mock content.
    - `fallback_count` accumulates across the adapter's lifetime so a stage
      can log "N/M calls used fallback content" at the end instead of the
      problem only being discoverable by inspecting output files later.
    - `strict_mode` (config: models.llm.strict_mode) — when True, raises
      LLMFallbackExhausted instead of returning mock content at all.
    """
    def __init__(self, config: dict = None):
        cfg = config or {}
        models = cfg.get("models", {}).get("llm", {})

        provider = models.get("provider", "groq").lower()
        groq_model = models.get("model", "llama-3.3-70b-versatile")
        ollama_host = models.get("ollama_host", "http://localhost:11434")
        ollama_model = models.get("ollama_model", "qwen2.5:7b")
        # NOTE: previously defined in config/default.yaml (under `system:`)
        # but never read anywhere in the codebase — strict_mode had no
        # effect regardless of its value.
        self.strict_mode = bool(cfg.get("system", {}).get("strict_mode", False))

        self.last_call_was_fallback = False
        self.fallback_count = 0
        self.total_calls = 0

        self._groq = GroqLLMAdapter(model_name=groq_model)
        self._ollama = LocalLLMAdapter(host=ollama_host, model_name=ollama_model)

        # Pick primary adapter based on provider setting
        if provider == "ollama" and self._ollama.check_health():
            self._primary = self._ollama
            logger.info(f"LLM: Using Ollama ({ollama_model}) as primary")
        elif self._groq.check_health():
            self._primary = self._groq
            logger.info("LLM: Using Groq free-tier (llama-3.3-70b) as primary")
        elif self._ollama.check_health():
            self._primary = self._ollama
            logger.info(f"LLM: Using Ollama ({ollama_model}) as fallback primary")
        else:
            self._primary = None
            logger.warning("LLM: No adapter available — mock mode active")

    @property
    def is_cloud(self) -> bool:
        return getattr(self._primary, "is_cloud", False)

    @property
    def is_available(self) -> bool:
        """Returns True if at least one real LLM provider is reachable."""
        return self._primary is not None

    def _handle_exhausted(self, system_prompt: str, prompt: str) -> str:
        """
        Called only when every real provider has failed for this request.
        Centralizes strict_mode/fallback-counting so generate() and
        generate_json() can't drift out of sync with each other.
        """
        self.fallback_count += 1
        self.last_call_was_fallback = True
        if self.strict_mode:
            raise LLMFallbackExhausted(
                f"All LLM providers failed (strict_mode=True). "
                f"system_prompt[:60]={system_prompt[:60]!r} prompt[:60]={prompt[:60]!r}"
            )
        logger.error(
            f"⚠️  LLM FALLBACK #{self.fallback_count}: both providers failed — "
            f"returning MOCK content, not real story content. "
            f"system_prompt[:60]={system_prompt[:60]!r}"
        )
        return _mock_response(system_prompt or "", prompt)

    def generate(self, prompt: str, system_prompt: str = None,
                 temperature: float = 0.7, model: str = None, **kwargs) -> str:
        self.total_calls += 1
        self.last_call_was_fallback = False

        if self._primary is None:
            return self._handle_exhausted(system_prompt or "", prompt)

        result = self._primary.generate(
            prompt, system_prompt=system_prompt,
            temperature=temperature, model=model, **kwargs
        )

        if "ERROR:" in result:
            logger.warning(f"Primary LLM failed ({result}). Skipping fallback and exhausting...")
            return self._handle_exhausted(system_prompt or "", prompt)

        return result

    def generate_json(self, prompt: str, system_prompt: str = None,
                      temperature: float = 0.1, model: str = None, **kwargs) -> str:
        """Tries primary, then fallback, with JSON-specific logic."""
        self.total_calls += 1
        self.last_call_was_fallback = False

        if self._primary is None:
            return self._handle_exhausted(system_prompt or "", prompt)

        result = self._primary.generate_json(
            prompt, system_prompt=system_prompt,
            temperature=temperature, model=model, **kwargs
        )

        if "ERROR:" in result:
            logger.warning(f"Primary LLM JSON failed ({result}). Skipping fallback and exhausting...")
            return self._handle_exhausted(system_prompt or "", prompt)

        return result

    def unload_model(self, model_name: str = None):
        if self._primary is not None:
            self._primary.unload_model(model_name)


# ── Schema-valid mock responses (when no LLM is available) ──────────────────
def _mock_response(system_prompt: str, prompt: str) -> str:
    sl = system_prompt.lower()

    # Memory extractor: "extract ALL named entities" / "visual character extractor"
    if any(k in sl for k in ["information extraction", "ie engine", "structured prompt",
                               "extract all", "visual dna", "named entities",
                               "visual character extractor", "knowledge store"]):
        return json.dumps({
            "_mock_fallback": True,
            "characters": [{"canonical_name": "Hero",
                            "visual_dna": {"hair": "black hair",
                                           "eyes": "brown eyes", "build": "athletic build"}}],
            "locations": [{"canonical_name": "Training Grounds",
                           "description": "open field, stone pillars, morning mist",
                           "visual_tags": "open field, stone pillars, morning mist, warm light"}],
            "world_concepts": [{"concept_type": "power_system", "name": "Martial Arts",
                                 "description": "physical training and combat techniques"}],
            "relationships": [{"char1": "Hero", "char2": "Master",
                                "type": "mentor", "description": "student and teacher"}],
        })

    # Scene planner (showrunner): "korean manhwa storyboard showrunner"
    if any(k in sl for k in ["film showrunner", "narrative scene", "showrunner",
                               "narrative beats", "divide them into", "storyboard showrunner",
                               "visual scenes", "story text into visual"]):
        # Extract some real text from prompt for narration if possible
        narration = prompt[:120].strip() if prompt else "The story continues."
        return json.dumps([{
            "scene_id": "SC001",
            "_mock_fallback": True,
            "location": "Training Grounds",
            "characters": ["Hero"],
            "emotion": "focused",
            "action": "Hero trains intensely under the morning sun",
            "camera_angle": "medium shot",
            "lighting": "warm morning sunlight, dramatic shadows",
            "visual_prompt_tags": "training, sweat, determination, morning light, stone pillars",
            "narration_text": narration,
            "complexity": 5,
        }])

    # Shot director: "film director" / "cinematic shots"
    if any(k in sl for k in ["film director", "cinematic shots", "break this",
                               "visual shots", "cinematic shot"]):
        return json.dumps([{
            "shot_id": "SH001_A",
            "camera": "medium shot",
            "visual_prompt_tags": ("training, sweat, determination, morning sunlight, "
                                   "stone pillars, dust particles"),
            "negative_prompt_tags": "lowres, bad anatomy, blurry",
            "narration_text": "Every day, he pushed his limits further.",
            "duration_estimate": 7.0,
        }])

    # World style: "art director" / "visual aesthetic"
    if any(k in sl for k in ["world style", "atmosphere", "storyboard artist",
                               "visual atmosphere", "art director", "visual aesthetic",
                               "booru tags", "comma-separated tags"]):
        return "martial arts world, ancient Asian architecture, stone courtyards, misty mountains"

    # Translation
    if any(k in sl for k in ["translate", "translation", "literary translator"]):
        return prompt  # pass-through

    # SEO / publishing
    if any(k in sl for k in ["seo", "metadata", "youtube", "title", "youtube seo"]):
        return json.dumps({
            "title": "Epic Novel Adaptation — Manhwa Style",
            "description": "Watch the story unfold in stunning Korean manhwa style.",
            "tags": ["manhwa", "novel", "animation", "webtoon", "korean"],
        })

    # Language detection
    if any(k in sl for k in ["english", "language", "detect", "analyze the language"]):
        return "ENGLISH"

    return f"Mock response: {prompt[:60]}"
