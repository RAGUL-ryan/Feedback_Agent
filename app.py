from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from agents.audio_agent import gtts_available, synthesize_gtts
from agents.translation_agent import get_gtts_params, supported_language_codes
from main import run_feedback_pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class FeedbackRequest(BaseModel):
    feedback: str
    input_mode: str | None = None
    voice_transcript: bool = False
    language: str = "en"          # "en" | "ta" | "hi" | "te"


class TTSRequest(BaseModel):
    text: str
    language: str = "en"          # controls gTTS voice language


@app.get("/")
def read_index():
    return FileResponse("index.html")


@app.get("/languages")
def get_languages():
    """Return supported language codes for the frontend dropdown."""
    return {"languages": supported_language_codes()}


@app.post("/analyze")
def analyze(request: FeedbackRequest):
    lang = request.language if request.language in supported_language_codes() else "en"
    result = run_feedback_pipeline(request.feedback, target_language=lang)
    result["input_mode"] = request.input_mode or "text"
    result["voice_transcript"] = request.voice_transcript
    result["audio_supported"] = gtts_available()
    return result


@app.post("/tts")
def generate_tts(request: TTSRequest):
    """
    Generate spoken audio for the given text.
    Language-specific gTTS params are resolved from translation_agent.
    """
    lang = request.language if request.language in supported_language_codes() else "en"
    gtts_lang, gtts_tld = get_gtts_params(lang)

    audio_bytes = synthesize_gtts(request.text, lang=gtts_lang, tld=gtts_tld)
    if not audio_bytes:
        raise HTTPException(
            status_code=503,
            detail="gTTS is unavailable. Install gTTS and ensure the server has internet access.",
        )
    return Response(content=audio_bytes, media_type="audio/mpeg")