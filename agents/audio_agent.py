from __future__ import annotations

from io import BytesIO
from typing import Optional

from config import GTTS_LANGUAGE
from config import GTTS_TLD


def gtts_available() -> bool:
    try:
        from gtts import gTTS  # noqa: F401
    except ImportError:
        return False
    return True


def synthesize_gtts(text: str) -> Optional[bytes]:
    if not text or not text.strip():
        return None

    try:
        from gtts import gTTS
    except ImportError:
        return None

    try:
        audio_buffer = BytesIO()
        tts = gTTS(text=text.strip(), lang=GTTS_LANGUAGE, tld=GTTS_TLD)
        tts.write_to_fp(audio_buffer)
        return audio_buffer.getvalue() or None
    except Exception:
        return None
