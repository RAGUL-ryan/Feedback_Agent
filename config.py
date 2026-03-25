import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.1-8b-instant"
EMBEDDING_MODEL = "text-embedding-3-small"
CHROMA_PERSIST_DIR = "./chroma_db"
ESCALATION_THRESHOLD = 0.3
GTTS_LANGUAGE = os.getenv("GTTS_LANGUAGE", "en")
GTTS_TLD = os.getenv("GTTS_TLD", "com")
