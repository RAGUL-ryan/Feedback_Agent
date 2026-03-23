import json
import re
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from config import MODEL_NAME

llm = ChatGroq(model=MODEL_NAME, temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a feedback ingestion agent. Analyze ANY type of feedback including emojis, star ratings, mixed text+emoji, or plain text.

Emoji meaning guide:
😡😤🤬💢 = very angry, complaint
😢💔😞😰 = sad, disappointed
😊❤️👍🌟😍✨ = happy, positive, praise
🤔❓🧐 = confused, has a question
🐛💥❌🔥 = bug report, technical issue
💰💳🧾 = billing issue
🚀⚡ = performance feedback
⭐ = star rating (count the stars)
👎 = negative, dissatisfied
👏🙌 = very positive, impressed

Return ONLY valid JSON, no markdown, no code fences:
{{
  "source": "text or emoji or mixed or rating",
  "original_text": "the original input exactly as given",
  "cleaned_text": "full human readable interpretation of what the user means",
  "detected_type": "text or emoji or mixed or rating",
  "emoji_sentiment": "positive or negative or neutral or mixed",
  "timestamp": null
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
            "detected_type": "text",
            "emoji_sentiment": "neutral",
            "timestamp": None
        }