"""
app.py
─────────────────────────────────────────────────────────────────────────────
v6 changes — Inline response mode (no CRM, no email):
  • Removed: email_agent and crm_agent imports (those files can be deleted)
  • /analyze response is now simpler — both 'replied' and 'escalated' return
    a `response` field that the UI renders directly in the feedback card.
  • Everything else (auth, JWT, TTS, language, sector) is unchanged.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from database import create_tables, get_db
from auth import router as auth_router, get_current_user
from agents.audio_agent import gtts_available, synthesize_gtts
from agents.translation_agent import get_gtts_params, supported_language_codes
from main import run_feedback_pipeline

app = FastAPI(title="Feedback Agent API")

# ── Create DB tables on startup ───────────────────────────────────────────────
@app.on_event("startup")
def startup():
    create_tables()
    print("[DB] Tables ready.")

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount auth routes ─────────────────────────────────────────────────────────
app.include_router(auth_router)


# ── Request models ────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    feedback: str
    input_mode: str | None = None
    voice_transcript: bool = False
    language: str = "en"
    sector: str | None = None


class TTSRequest(BaseModel):
    text: str
    language: str = "en"


# ── Public routes ─────────────────────────────────────────────────────────────

@app.get("/")
def read_index():
    return FileResponse("index.html")


@app.get("/languages")
def get_languages():
    return {"languages": supported_language_codes()}


# ── Protected route: /analyze requires login ──────────────────────────────────

@app.post("/analyze")
def analyze(
    request: FeedbackRequest,
    current_user=Depends(get_current_user),
):
    """
    Process customer feedback and return an inline response.
    Both 'replied' and 'escalated' statuses return a `response` field
    that is displayed directly in the UI.
    """
    lang = request.language if request.language in supported_language_codes() else "en"

    result = run_feedback_pipeline(
        raw_feedback    = request.feedback,
        target_language = lang,
        sector          = request.sector,
        customer_name   = current_user.full_name,
        customer_email  = current_user.email,
        customer_phone  = current_user.phone or "",
    )

    result["input_mode"]       = request.input_mode or "text"
    result["voice_transcript"] = request.voice_transcript
    result["audio_supported"]  = gtts_available()
    return result


@app.post("/tts")
def generate_tts(request: TTSRequest):
    lang = request.language if request.language in supported_language_codes() else "en"
    gtts_lang, gtts_tld = get_gtts_params(lang)
    audio_bytes = synthesize_gtts(request.text, lang=gtts_lang, tld=gtts_tld)
    if not audio_bytes:
        raise HTTPException(status_code=503, detail="gTTS unavailable.")
    return Response(content=audio_bytes, media_type="audio/mpeg")