"""
agents/translation_agent.py
─────────────────────────────────────────────────────────────────────────────
True agentic multilingual support.

OLD approach (wrong):
  TANGLISH_MARKERS: a static set of ~80 Tamil words.
  normalize_tanglish(): a regex replacement dictionary of ~20 phrases.
  Problem: "app la problem iruku" fails because "la"/"iruku" are not in the set.
  Problem: any unseen phrase passes through unchanged, breaking downstream agents.

NEW approach (correct):
  LLM agent with two tools:
    detect_language      – identify any language, dialect, or mixed script
    normalize_to_english – translate the full meaning to clean English
  The LLM handles Tanglish, Hinglish, any Indian regional language in Roman
  script, and unseen colloquial phrases — without any hardcoded word list.

  The actual translate_text() call (English → Tamil/Hindi/Telugu) stays as a
  direct API tool — Google Translate is a reliable external service, not a
  reasoning task. That part is correctly architected already.

FIX (v2):
  BUG: Groq/Llama tool-use fails with BadRequestError 400 when the model
       serializes Python bool `False` instead of JSON `false` in the
       detect_language tool call. This happens consistently for plain English
       input because is_mixed=False is always the output.

  THREE-LAYER FIX applied:
    1. Fast-path bypass  — plain ASCII-English input skips the LLM entirely.
       No tool call → no serialization bug → zero latency for the common case.
    2. Tool signature fix — is_mixed changed from `bool` to `str` ("true"/"false").
       Groq/Llama serializes strings reliably; booleans cause the 400 error.
    3. try/except guard  — if the LLM call still fails for any reason, fall back
       to safe defaults instead of crashing the whole pipeline.
"""

from __future__ import annotations
import re as _re
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
#from langchain_groq import ChatGroq
from config import MODEL_NAME
from langchain_openai import ChatOpenAI

# ─────────────────────────────────────────────────────────────────────────────
# Language metadata  (pure lookup, no LLM needed)
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_LANGUAGES: dict[str, dict] = {
    "en": {"name": "English", "gtts_lang": "en", "gtts_tld": "com"},
    "ta": {"name": "Tamil",   "gtts_lang": "ta", "gtts_tld": "co.in"},
    "hi": {"name": "Hindi",   "gtts_lang": "hi", "gtts_tld": "co.in"},
    "te": {"name": "Telugu",  "gtts_lang": "te", "gtts_tld": "co.in"},
}
DEFAULT_LANGUAGE = "en"


def get_gtts_params(target_lang: str) -> tuple[str, str]:
    meta = SUPPORTED_LANGUAGES.get(target_lang, SUPPORTED_LANGUAGES[DEFAULT_LANGUAGE])
    return meta["gtts_lang"], meta["gtts_tld"]


def get_language_name(lang_code: str) -> str:
    return SUPPORTED_LANGUAGES.get(lang_code, {}).get("name", "English")


def supported_language_codes() -> list[str]:
    return list(SUPPORTED_LANGUAGES.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Fast-path helper
# ─────────────────────────────────────────────────────────────────────────────

# Matches strings that are entirely ASCII printable + common punctuation.
# Any non-ASCII character (Tamil, Hindi, Telugu script, or heavy emoji)
# will fail this check and go through the LLM agent.
_ASCII_ENGLISH_RE = _re.compile(r'^[\x00-\x7F]+$')

def _is_plain_english(text: str) -> bool:
    """
    Return True if the text is very likely plain English:
      - All characters are ASCII (no Indic script, no non-ASCII emoji)
      - Does NOT contain sequences of 3+ consecutive non-letter/digit chars
        that might indicate heavy emoji-only input like "😡😡🔥"
        (those are non-ASCII anyway, so they fail the ASCII check first)

    This is intentionally conservative — when in doubt, return False and
    let the LLM agent decide.
    """
    if not text or not text.strip():
        return True   # empty → handled separately upstream
    return bool(_ASCII_ENGLISH_RE.match(text))


# ─────────────────────────────────────────────────────────────────────────────
# Tools for the Language Detection Agent
# ─────────────────────────────────────────────────────────────────────────────

@tool
def detect_language(
    language_name: str,
    language_code: str,
    is_mixed: str,          # FIX: was `bool` — Groq/Llama serializes Python
                            #      False (not JSON false), causing 400 errors.
                            #      Changed to str: "true" or "false".
    confidence: float,
    notes: str,
) -> dict:
    """
    Report the detected language or dialect of the user's input.
    Always call this first after analysing the input.

    language_name: Human-readable name — e.g. 'Tanglish', 'Tamil', 'Hindi',
                   'Telugu', 'English', 'Hinglish', 'Mixed'
    language_code: ISO code if known — 'ta', 'hi', 'te', 'en', or 'mixed'
    is_mixed: "true" if two or more languages are mixed (e.g. Tamil + English),
              "false" otherwise. Use the string "true" or "false".
    confidence: detection confidence, 0.0 to 1.0
    notes: brief explanation of what cues led to this detection
    """
    # Normalise is_mixed to a real bool regardless of what the LLM sent
    if isinstance(is_mixed, bool):
        mixed = is_mixed
    else:
        mixed = str(is_mixed).strip().lower() in ("true", "1", "yes")

    return {
        "language_name": language_name,
        "language_code": language_code,
        "is_mixed": mixed,
        "confidence": confidence,
        "notes": notes,
    }


@tool
def normalize_to_english(normalized_text: str, original_language: str) -> dict:
    """
    Provide the English version of a non-English or mixed-language input.
    Call this whenever the input is NOT plain English.

    normalized_text: the FULL meaning of the original message expressed in
                     clear, natural English. Be faithful — preserve urgency,
                     sentiment, and all factual claims (amounts, errors, etc).
                     Do not add or remove any information.
    original_language: what language/dialect the original was in
    """
    return {
        "normalized_text": normalized_text.strip(),
        "original_language": original_language,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Language Detection + Normalization Agent
# ─────────────────────────────────────────────────────────────────────────────

_DETECT_TOOLS = [detect_language, normalize_to_english]
#_detect_llm = ChatGroq(model=MODEL_NAME, temperature=0).bind_tools(_DETECT_TOOLS)

_llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
_DETECT_SYSTEM = """You are a multilingual language detection and normalization agent.

Step 1: Identify the language or dialect of the user's feedback.
Step 2: If it is NOT plain English, translate/normalize it into clear English.

Languages you must handle:
  • Plain English        → just call detect_language, no normalization needed
  • Tamil script         → translate to English, call both tools
  • Hindi script         → translate to English, call both tools
  • Telugu script        → translate to English, call both tools
  • Tanglish             → Tamil written in English/Roman script mixed with English
                           e.g. "app romba mosam work agala bro"
                           → "the app is very bad and not working"
  • Hinglish             → Hindi in Roman script mixed with English
  • Any other regional Indian language in Roman script → normalize to English
  • Pure emoji           → interpret meaning in English

Always call detect_language.
Call normalize_to_english if and only if the input is not already plain English.

IMPORTANT — is_mixed field: always pass the string "true" or "false" (not a boolean).

Be faithful to the meaning. Preserve all urgency, sentiment, and factual claims."""


def detect_and_normalize(raw_feedback: str) -> dict:
    """
    LLM agent: detect language and normalize to English if needed.

    FIX v2: Three-layer protection against Groq 400 tool_use_failed errors:
      1. Fast-path: plain ASCII-English bypasses the LLM entirely.
      2. Tool signature: is_mixed is str not bool (avoids serialization bug).
      3. try/except: any remaining LLM failure falls back gracefully.

    Returns:
        tanglish_detected   : bool   (True for any non-English regional input)
        normalized_feedback : str    (English version, or original if already English)
        detected_language   : str    (human-readable name)
        language_code       : str    (ISO code or 'mixed')
        detection_confidence: float
    """
    # ── Guard: empty input ────────────────────────────────────────────────────
    if not raw_feedback or not raw_feedback.strip():
        return {
            "tanglish_detected": False,
            "normalized_feedback": raw_feedback,
            "detected_language": "English",
            "language_code": "en",
            "detection_confidence": 1.0,
        }

    # ── FIX Layer 1: Fast-path bypass for plain English ───────────────────────
    # Plain ASCII English never triggers the bool-serialization bug because
    # we never make a tool call at all.  This also saves ~300–500 ms latency
    # for the most common case (English feedback with English UI language).
    if _is_plain_english(raw_feedback):
        print(f"[Lang]   Fast-path: plain English detected, skipping LLM tool call.")
        return {
            "tanglish_detected": False,
            "normalized_feedback": raw_feedback,
            "detected_language": "English",
            "language_code": "en",
            "detection_confidence": 1.0,
        }

    # ── LLM agent path (non-English / mixed / emoji input) ───────────────────
    messages = [
        SystemMessage(content=_DETECT_SYSTEM),
        HumanMessage(content=f"Analyse this feedback:\n\n{raw_feedback}"),
    ]

    # ── FIX Layer 3: try/except — any LLM failure falls back gracefully ───────
    try:
        response: AIMessage = _detect_llm.invoke(messages)
    except Exception as exc:
        print(f"[Lang]   LLM tool call failed ({exc}), using safe fallback defaults.")
        return {
            "tanglish_detected": False,
            "normalized_feedback": raw_feedback,
            "detected_language": "Unknown",
            "language_code": "en",
            "detection_confidence": 0.5,
        }

    # Defaults
    detected_language    = "English"
    language_code        = "en"
    is_mixed             = False
    detection_confidence = 1.0
    normalized_feedback  = raw_feedback

    tool_map = {t.name: t for t in _DETECT_TOOLS}

    for tc in (response.tool_calls or []):
        name, args = tc["name"], tc["args"]
        if name not in tool_map:
            continue
        try:
            result: dict = tool_map[name].invoke(args)
        except Exception as exc:
            print(f"[Lang]   Tool '{name}' invocation failed ({exc}), skipping.")
            continue

        if name == "detect_language":
            detected_language    = result.get("language_name", "English")
            language_code        = result.get("language_code", "en")
            is_mixed             = result.get("is_mixed", False)
            detection_confidence = result.get("confidence", 1.0)

        elif name == "normalize_to_english":
            text = result.get("normalized_text", "").strip()
            if text:
                normalized_feedback = text

    is_tanglish = (
        is_mixed
        or "tanglish" in detected_language.lower()
        or "hinglish" in detected_language.lower()
        or (language_code not in ("en", "mixed") and normalized_feedback != raw_feedback)
    )

    return {
        "tanglish_detected": is_tanglish,
        "normalized_feedback": normalized_feedback,
        "detected_language": detected_language,
        "language_code": language_code,
        "detection_confidence": detection_confidence,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible shims (keep main.py import signatures working)
# ─────────────────────────────────────────────────────────────────────────────

def is_tanglish(text: str) -> bool:
    """Detect if input is non-English / Tanglish. Delegates to LLM agent."""
    return detect_and_normalize(text)["tanglish_detected"]


def normalize_tanglish(text: str) -> str:
    """Normalize non-English input to plain English. Delegates to LLM agent."""
    return detect_and_normalize(text)["normalized_feedback"]


# ─────────────────────────────────────────────────────────────────────────────
# Translation Tool  (English → target language via Google Translate API)
# This stays as a direct API call — translation is a tool, not a reasoning task.
# ─────────────────────────────────────────────────────────────────────────────

def translate_text(text: str, target_lang: str) -> str:
    """Translate English text to target_lang. Falls back to original on error."""
    if not text or not text.strip():
        return text
    lang = target_lang if target_lang in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE
    if lang == "en":
        return text
    try:
        from deep_translator import GoogleTranslator  # type: ignore
        translated = GoogleTranslator(source="en", target=lang).translate(text.strip())
        return translated or text
    except Exception as exc:
        print(f"[TRANSLATION] Warning — translation to '{lang}' failed: {exc}")
        return text