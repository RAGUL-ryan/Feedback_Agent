"""
agents/ingestion_agent.py
─────────────────────────────────────────────────────────────────────────────
Feedback ingestion agent. First contact with raw user input.
Interprets ANY format and returns a clean normalized representation.

Improvements over original:
  • Handles sarcasm explicitly
  • Extracts star_rating as a numeric field
  • cleaned_text must be a faithful full English sentence, not a copy of the original
"""

import json
import re
from langchain_core.prompts import ChatPromptTemplate

from config import MODEL_NAME
from langchain_openai import ChatOpenAI

_llm = ChatOpenAI(model=MODEL_NAME, temperature=0)

_SYSTEM = """You are a feedback ingestion agent for a customer support system.

Receive raw customer feedback in ANY format and produce a clean, structured interpretation.

Emoji reference:
  😡 😤 🤬 💢  → very angry, complaint
  😢 💔 😞 😰  → sad, disappointed
  😊 ❤️ 👍 🌟 😍 ✨ → happy, positive
  🤔 ❓ 🧐     → confused, questioning
  🐛 💥 ❌ 🔥  → bug, technical issue
  💰 💳 🧾     → billing
  ⭐           → star rating (count total stars)
  👎           → negative
  👏 🙌        → very positive

Star rating:
  1 star = extremely negative, 2 = negative, 3 = neutral, 4 = positive, 5 = very positive

Important:
  • For sarcasm (e.g. "great, crashed again") → cleaned_text reflects the REAL negative meaning
  • cleaned_text must be a complete English sentence explaining what the user means
  • Never truncate or omit factual claims (amounts, error messages, dates)
  • star_rating: integer 1–5 if detectable, else null

Return ONLY valid JSON — no markdown, no fences:
{{
  "source": "text | emoji | mixed | rating",
  "original_text": "<exact original>",
  "cleaned_text": "<full faithful English interpretation>",
  "detected_type": "text | emoji | mixed | rating",
  "emoji_sentiment": "positive | negative | neutral | mixed",
  "star_rating": null,
  "timestamp": null
}}"""

_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human", "{raw_feedback}"),
])


def ingest(raw_feedback: str) -> dict:
    chain = _prompt | _llm
    result = chain.invoke({"raw_feedback": raw_feedback})
    text = re.sub(r"```json|```", "", result.content.strip()).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "source": "unknown",
            "original_text": raw_feedback,
            "cleaned_text": raw_feedback,
            "detected_type": "text",
            "emoji_sentiment": "neutral",
            "star_rating": None,
            "timestamp": None,
        }