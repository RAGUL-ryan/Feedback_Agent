from rag.vectorstore import add_to_vectorstore

def learn(feedback: str, response: str, outcome: str = "auto_resolved"):
    """
    Stores successful feedback+response pairs back into the vector DB.
    This improves future RAG context retrieval.
    """
    text = f"Feedback: {feedback}\nResolution: {response}"
    metadata = {"outcome": outcome, "source": "learning_agent"}
    add_to_vectorstore(text, metadata)
    print(f"[LEARNING] New case stored to knowledge base.")
