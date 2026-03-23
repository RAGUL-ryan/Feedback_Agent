import json
import re
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from config import MODEL_NAME

llm = ChatGroq(model=MODEL_NAME, temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a feedback understanding agent. Analyze feedback and return ONLY a valid JSON object with no extra text, no markdown, no code fences.
Return exactly this structure:
{{
  "sentiment": "positive or neutral or negative",
  "intent": "complaint or praise or question or suggestion or bug_report",
  "topic": "billing or product or support or delivery or general",
  "urgency": "low or medium or high",
  "confidence": 0.8
}}"""),
    ("human", "{cleaned_feedback}")
])

def understand(cleaned_feedback: str) -> dict:
    chain = prompt | llm
    result = chain.invoke({"cleaned_feedback": cleaned_feedback})
    text = result.content.strip()
    # Strip markdown code fences if present
    text = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback default if parsing still fails
        return {
            "sentiment": "negative",
            "intent": "complaint",
            "topic": "general",
            "urgency": "medium",
            "confidence": 0.3
        }