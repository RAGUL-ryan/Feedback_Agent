from __future__ import annotations

from io import BytesIO
from typing import Optional

from config import GTTS_LANGUAGE, GTTS_TLD


def gtts_available() -> bool:
    try:
        from gtts import gTTS  # noqa: F401
    except ImportError:
        return False
    return True


def synthesize_gtts(
    text: str,
    lang: str | None = None,
    tld: str | None = None,
) -> Optional[bytes]:
    if not text or not text.strip():
        return None

    try:
        from gtts import gTTS
    except ImportError:
        return None

    effective_lang = lang or GTTS_LANGUAGE
    effective_tld = tld or GTTS_TLD

    try:
        audio_buffer = BytesIO()
        tts = gTTS(text=text.strip(), lang=effective_lang, tld=effective_tld)
        tts.write_to_fp(audio_buffer)
        return audio_buffer.getvalue() or None
    except Exception as exc:
        print(f"[AUDIO] gTTS error (lang={effective_lang}, tld={effective_tld}): {exc}")
        return None