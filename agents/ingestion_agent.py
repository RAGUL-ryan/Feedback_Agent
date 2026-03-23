import json
import re
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from config import MODEL_NAME

llm = ChatGroq(model=MODEL_NAME, temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a feedback ingestion agent. Clean and normalize the feedback.
Return ONLY a valid JSON object with no markdown, no code fences, no extra text.
Return exactly this structure:
{{
  "source": "email or chat or form or unknown",
  "original_text": "the original feedback",
  "cleaned_text": "cleaned and normalized version",
  "timestamp": "if present or null"
}}"""),
    ("human", "{raw_feedback}")
])

def ingest(raw_feedback: str) -> dict:
    chain = prompt | llm
    result = chain.invoke({"raw_feedback": raw_feedback})
    text = result.content.strip()
    text = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "source": "unknown",
            "original_text": raw_feedback,
            "cleaned_text": raw_feedback,
            "timestamp": None
        }