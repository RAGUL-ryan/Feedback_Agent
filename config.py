import os
from dotenv import load_dotenv

load_dotenv()

KNOWLEDGE_BASE_DIR = "./data/knowledge_base"
#GROQ_API_KEY       = os.getenv("GROQ_API_KEY")
#MODEL_NAME         = "llama-3.3-70b-versatile"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME     = "gpt-4o-mini" 
SECTOR             = "fintech"
EMBEDDING_MODEL    = "text-embedding-3-small"
CHROMA_PERSIST_DIR = "./chroma_db"
KNOWLEDGE_BASE_CHUNK_SIZE    = 2000
KNOWLEDGE_BASE_CHUNK_OVERLAP = 200
ESCALATION_THRESHOLD         = 0.3
GTTS_LANGUAGE = os.getenv("GTTS_LANGUAGE", "en")
GTTS_TLD      = os.getenv("GTTS_TLD", "com")