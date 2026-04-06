"""
rag/embeddings.py

OPTIMIZATION: Cache the HuggingFaceEmbeddings instance as a module-level singleton.

OLD: get_embeddings() created a new HuggingFaceEmbeddings object on every call.
     This re-initializes the sentence-transformer model each time — ~500ms–2s overhead
     on the first call per request if the object isn't reused.

NEW: The model is loaded once on first call and reused for all subsequent calls.
     Zero initialization overhead from the second call onward.
"""

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Module-level singleton — initialized once, reused forever
_embeddings_instance: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings_instance
    if _embeddings_instance is None:
        print("[Embeddings] Loading HuggingFace model (first call only)...")
        _embeddings_instance = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        print("[Embeddings] Model loaded and cached.")
    return _embeddings_instance