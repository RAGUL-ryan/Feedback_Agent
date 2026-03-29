"""
agents/translation_agent.py
────────────────────────────────────────────────────────────────────────────
Multilingual support for the Feedback Agent pipeline.

Features
────────
• Translate any response text into Tamil, Hindi, or Telugu via Google Translate
  (free — no API key required, uses deep-translator under the hood).
• Detect "Tanglish" (Tamil written in English / Roman script) and normalise it
  so the rest of the pipeline processes it correctly.
• Returns the original English text unchanged when the target language is "en".

Dependencies
────────────
    pip install deep-translator

Supported language codes
────────────────────────
    "en"  → English  (no translation — passthrough)
    "ta"  → Tamil    (தமிழ்)
    "hi"  → Hindi    (हिन्दी)
    "te"  → Telugu   (తెలుగు)
"""

from __future__ import annotations

import re
from typing import Optional

# ── Tanglish word bank ────────────────────────────────────────────────────────
# Common Tamil words/phrases written in English (Roman script).
# These are checked against the user's input to detect Tanglish.
TANGLISH_MARKERS: set[str] = {
    # Pronouns / particles
    "naan", "nee", "avan", "aval", "naam", "ungaluku", "enaku", "avanga",
    "inga", "anda", "enna", "epdi", "eppadi", "yenna", "yepdi",
    # Common verbs
    "panrom", "pannrom", "panna", "seiyanum", "sollunga", "parunga",
    "vandhutaen", "vandhen", "poren", "porom", "iruku", "irukku",
    "theriyala", "therila", "puriyala", "purila", "vendum", "vendam",
    "mudiyala", "mudila", "aagala", "agala",
    # Common nouns / adjectives
    "padam", "kadai", "vilaiyattu", "vilayaddu", "saapdu", "saapadu",
    "pasikuthu", "thanni", "amma", "appa", "anna", "akka", "thambi", "thangachi",
    "kalyanam", "veetu", "veedu", "ooru", "oor",
    # Common expressions
    "super", "romba", "romba nalla", "nalla", "mosam", "mela", "kila",
    "thapu", "seri", "seri da", "da", "di", "bro", "machan", "machi",
    "adhu", "idhu", "ithu", "athu", "appuram", "aprom",
    # Billing / support context
    "charge pannitanga", "charge pannanga", "refund kudunga", "help pannga",
    "app work agala", "app velaikagala", "password maranthutten",
    "twice charge", "double charge pannitanga",
}

# ── Language metadata ─────────────────────────────────────────────────────────
SUPPORTED_LANGUAGES: dict[str, dict] = {
    "en": {"name": "English",   "gtts_lang": "en", "gtts_tld": "com"},
    "ta": {"name": "Tamil",     "gtts_lang": "ta", "gtts_tld": "co.in"},
    "hi": {"name": "Hindi",     "gtts_lang": "hi", "gtts_tld": "co.in"},
    "te": {"name": "Telugu",    "gtts_lang": "te", "gtts_tld": "co.in"},
}

DEFAULT_LANGUAGE = "en"


# ── Tanglish detector ─────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lower-case, strip punctuation, split into tokens."""
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return cleaned.split()


def is_tanglish(text: str) -> bool:
    """
    Return True if the input looks like Tanglish (Tamil written in English).

    Heuristic: if ≥2 tokens match known Tanglish words, or a well-known
    multi-word phrase is present, classify as Tanglish.
    """
    if not text:
        return False

    tokens = set(_tokenize(text))
    hit_count = len(tokens & TANGLISH_MARKERS)

    # Check multi-word phrases too
    lower = text.lower()
    phrase_hits = sum(
        1 for phrase in TANGLISH_MARKERS
        if " " in phrase and phrase in lower
    )

    return (hit_count + phrase_hits) >= 2


def normalize_tanglish(text: str) -> str:
    """
    A lightweight Tanglish → plain English normalisation pass.
    This lets the understanding agent process the semantic meaning
    without being confused by Tamil vocabulary.

    For production, replace this with a dedicated Tanglish NLP model
    (e.g. fine-tuned mBART or IndicBERT).
    """
    replacements: dict[str, str] = {
        # Sentiment / urgency words
        r"\bromba\b": "very",
        r"\bnalla\b": "good",
        r"\bmosam\b": "bad",
        r"\bseri\b": "okay",
        r"\bthapu\b": "wrong",
        r"\bsuperr?\b": "excellent",
        # Common phrases
        r"\bcharge pannitanga\b": "charged me",
        r"\bcharge pannanga\b": "they charged",
        r"\bdouble charge pannitanga\b": "charged twice",
        r"\btwice charge\b": "charged twice",
        r"\brefund kudunga\b": "please give refund",
        r"\bhelp pannga\b": "please help",
        r"\bapp work agala\b": "app is not working",
        r"\bapp velaikagala\b": "app is not working",
        r"\bpassword maranthutten\b": "I forgot my password",
        r"\bpuriyala\b": "I don't understand",
        r"\bpurila\b": "I don't understand",
        r"\btheriyala\b": "I don't know",
        r"\btherila\b": "I don't know",
        r"\bmudiyala\b": "unable to",
        r"\bmudila\b": "unable to",
        r"\bpasikuthu\b": "hungry",
        # Filler / particles (just remove)
        r"\b(da|di|bro|machan|machi|yaar|pa)\b": "",
    }

    result = text
    for pattern, replacement in replacements.items():
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    # Collapse extra whitespace
    return " ".join(result.split()).strip()


# ── Translation ───────────────────────────────────────────────────────────────

def translate_text(text: str, target_lang: str) -> str:
    """
    Translate *text* into *target_lang* using Google Translate (free tier).

    Falls back to the original English text on any error so the pipeline
    never breaks because of a translation failure.

    Args:
        text:        The English text to translate.
        target_lang: One of "en", "ta", "hi", "te".

    Returns:
        Translated string, or original string if translation fails / lang=="en".
    """
    if not text or not text.strip():
        return text

    lang = target_lang if target_lang in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE

    if lang == "en":
        return text  # no-op

    try:
        from deep_translator import GoogleTranslator  # type: ignore
        translated = GoogleTranslator(source="en", target=lang).translate(text.strip())
        return translated or text
    except Exception as exc:
        print(f"[TRANSLATION] Warning — translation to '{lang}' failed: {exc}")
        return text  # graceful fallback


# ── gTTS language helper ──────────────────────────────────────────────────────

def get_gtts_params(target_lang: str) -> tuple[str, str]:
    """
    Return (gtts_lang, gtts_tld) for the given language code.
    Used by audio_agent to speak in the correct language.
    """
    meta = SUPPORTED_LANGUAGES.get(target_lang, SUPPORTED_LANGUAGES[DEFAULT_LANGUAGE])
    return meta["gtts_lang"], meta["gtts_tld"]


# ── Public helpers ────────────────────────────────────────────────────────────

def get_language_name(lang_code: str) -> str:
    """Human-readable language name for a given code."""
    return SUPPORTED_LANGUAGES.get(lang_code, {}).get("name", "English")


def supported_language_codes() -> list[str]:
    return list(SUPPORTED_LANGUAGES.keys())