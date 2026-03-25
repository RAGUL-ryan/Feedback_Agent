from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import Response
from pydantic import BaseModel

from agents.audio_agent import gtts_available
from agents.audio_agent import synthesize_gtts
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


class TTSRequest(BaseModel):
    text: str


@app.get("/")
def read_index():
    return FileResponse("index.html")

@app.post("/analyze")
def analyze(request: FeedbackRequest):
    result = run_feedback_pipeline(request.feedback)
    result["input_mode"] = request.input_mode or "text"
    result["voice_transcript"] = request.voice_transcript
    result["audio_supported"] = gtts_available()
    return result


@app.post("/tts")
def generate_tts(request: TTSRequest):
    audio_bytes = synthesize_gtts(request.text)
    if not audio_bytes:
        raise HTTPException(
            status_code=503,
            detail="gTTS is unavailable. Install gTTS and ensure the server has internet access.",
        )
    return Response(content=audio_bytes, media_type="audio/mpeg")
