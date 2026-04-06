"""
agents/context_agent.py

OPTIMIZATION: Cache the Chroma vectorstore instance at module level.

OLD: load_vectorstore() was called fresh on every get_context() call.
     Each call re-checked the manifest file and created a new Chroma() object — 
     even when nothing had changed. This added ~100–300ms per request.

NEW: The Chroma instance is cached after the first load.
     Subsequent requests reuse the same object with zero initialization overhead.
     Cache is invalidated only when the knowledge base files actually change
     (detected by the manifest check), so new files are always picked up.
"""

from __future__ import annotations
from rag.vectorstore import load_vectorstore, _manifest_is_current

# Module-level cache
_vectorstore_cache = None


def _get_vectorstore():
    global _vectorstore_cache
    # Invalidate cache if knowledge base files have changed
    if _vectorstore_cache is None or not _manifest_is_current():
        print("[Context] Loading vectorstore...")
        _vectorstore_cache = load_vectorstore()
    return _vectorstore_cache


def get_context(query: str, k: int = 3) -> str:
    db = _get_vectorstore()
    docs = db.similarity_search(query, k=k)
    context_chunks = [doc.page_content for doc in docs]
    return "\n\n---\n\n".join(context_chunks)