import json
import re
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from config import MODEL_NAME

llm = ChatGroq(model=MODEL_NAME, temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a feedback understanding agent. Analyze any feedback — text, emojis, ratings, or mixed.

Emoji interpretation:
😡😤🤬💢 → negative, complaint, high urgency
😢💔😞😰 → negative, disappointment, medium urgency
😊❤️👍🌟😍 → positive, praise, low urgency
🤔❓🧐 → neutral, question, low urgency
🐛💥🔥❌ → negative, bug_report, high urgency
👎 → negative, complaint
👏🙌 → positive, praise
⭐ count: 1=very negative, 2=negative, 3=neutral, 4=positive, 5=very positive

Return ONLY valid JSON, no markdown, no code fences:
{{
  "sentiment": "positive or neutral or negative",
  "intent": "complaint or praise or question or suggestion or bug_report",
  "topic": "billing or product or support or delivery or general",
  "urgency": "low or medium or high",
  "confidence": 0.8,
  "input_type": "text or emoji or mixed or rating",
  "emoji_detected": true,
  "human_readable": "one sentence plain english summary of what the user is saying"
}}"""),
    ("human", "{cleaned_feedback}")
])

def understand(cleaned_feedback: str) -> dict:
    chain = prompt | llm
    result = chain.invoke({"cleaned_feedback": cleaned_feedback})
    text = result.content.strip()
    text = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "sentiment": "neutral",
            "intent": "general",
            "topic": "general",
            "urgency": "low",
            "confidence": 0.5,
            "input_type": "text",
            "emoji_detected": False,
            "human_readable": cleaned_feedback
        }