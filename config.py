import os
from dotenv import load_dotenv

load_dotenv()

KNOWLEDGE_BASE_DIR = "./data/knowledge_base"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# FIX: llama-3.1-8b-instant is too small for reliable tool-use.
# It generates tool calls in a broken XML format (<function=name>{args})
# instead of proper JSON, causing Groq 400 BadRequestError on every
# tool-calling agent (translation_agent, decision_agent, edge_case_agent).
# llama-3.3-70b-versatile supports tool-use correctly and is still free on Groq.
MODEL_NAME = "llama-3.3-70b-versatile"

# ─────────────────────────────────────────────────────────────────────────────
# ADD THIS TO YOUR config.py
# ─────────────────────────────────────────────────────────────────────────────

# Sector for your deployment.
# Options: "fintech" | "education" | "healthcare" | "food" | "ecommerce"
SECTOR = "fintech"

# You can also override sector at runtime per-request in main.py:
#   result = pipeline.run(feedback, sector="healthcare")
# Each agent accepts an optional sector= param that takes priority over this.

EMBEDDING_MODEL = "text-embedding-3-small"
CHROMA_PERSIST_DIR = "./chroma_db"
KNOWLEDGE_BASE_CHUNK_SIZE = 2000
KNOWLEDGE_BASE_CHUNK_OVERLAP = 200
ESCALATION_THRESHOLD = 0.3
GTTS_LANGUAGE = os.getenv("GTTS_LANGUAGE", "en")
GTTS_TLD = os.getenv("GTTS_TLD", "com")