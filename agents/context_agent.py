from rag.vectorstore import load_vectorstore

def get_context(query: str, k: int = 3) -> str:
    db = load_vectorstore()
    docs = db.similarity_search(query, k=k)
    context_chunks = [doc.page_content for doc in docs]
    return "\n\n---\n\n".join(context_chunks)
