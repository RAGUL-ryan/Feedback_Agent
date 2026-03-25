import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from config import MODEL_NAME

llm = ChatGroq(model=MODEL_NAME, temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a professional feedback understanding agent for a production customer-feedback system.

Your job is to classify the customer's real business meaning, not just surface wording.

Guidelines:
- Handle direct feedback, mixed feedback, subtle complaints, disappointment, sarcasm, slang, shorthand, emojis, ratings, and imperfect voice transcripts.
- If a sentence starts positive but ends with failure or dissatisfaction, prioritize the final customer outcome.
- Do not over-trust words like "good", "nice", or "great" when the message also says the product broke, failed, or caused pain.
- Sentiment is only one field. Also classify intent, severity, urgency, and whether a human should review it.
- Use human review only when risk, ambiguity, or business impact justifies it.

Return ONLY valid JSON:
{{
  "sentiment": "positive or neutral or negative",
  "intent": "complaint or praise or question or suggestion or bug_report",
  "topic": "billing or product or support or delivery or general",
  "severity": "low or medium or high",
  "urgency": "low or medium or high",
  "needs_human_review": false,
  "review_reason": "",
  "confidence": 0.82,
  "input_type": "text or emoji or mixed or rating",
  "emoji_detected": false,
  "human_readable": "one sentence plain english summary of what the user is saying"
}}""",
        ),
        ("human", "{cleaned_feedback}"),
    ]
)

VALID_SENTIMENTS = {"positive", "neutral", "negative"}
VALID_INTENTS = {"complaint", "praise", "question", "suggestion", "bug_report"}
VALID_TOPICS = {"billing", "product", "support", "delivery", "general"}
VALID_LEVELS = {"low", "medium", "high"}

HIGH_RISK_TERMS = {
    "fraud",
    "scam",
    "hacked",
    "hack",
    "security",
    "compromised",
    "data leak",
    "charged twice",
    "double charged",
}

QUESTION_TERMS = {"how", "what", "why", "where", "when", "can", "could", "help"}
BUG_TERMS = {"broken", "broke", "crash", "crashed", "bug", "error", "failed", "not working", "stuck"}
BILLING_TERMS = {"charged", "charge", "refund", "payment", "invoice", "billing", "subscription"}
DELIVERY_TERMS = {"delivery", "shipping", "order", "package", "dispatch", "tracking"}
SUPPORT_TERMS = {"support", "agent", "reply", "response", "wait"}


def _normalize_text(text: str) -> str:
    normalized = (text or "").strip().lower()
    replacements = {
        "asap": "as soon as possible",
        "pls": "please",
        "plz": "please",
        "thx": "thanks",
        "idk": "i do not know",
        "wtf": "this is unacceptable",
        "omg": "surprised",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return " ".join(normalized.split())


def _contains_any(text: str, phrases: set[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def _fallback_understanding(text: str) -> dict:
    normalized = _normalize_text(text)

    intent = "praise"
    topic = "general"
    sentiment = "neutral"
    severity = "low"
    urgency = "low"
    needs_human_review = False
    review_reason = ""

    if _contains_any(normalized, BUG_TERMS):
        intent = "bug_report"
        topic = "product"
        sentiment = "negative"
        severity = "medium"
        urgency = "medium"
    elif _contains_any(normalized, BILLING_TERMS):
        intent = "complaint"
        topic = "billing"
        sentiment = "negative"
        severity = "medium"
        urgency = "medium"
    elif _contains_any(normalized, DELIVERY_TERMS):
        topic = "delivery"
    elif _contains_any(normalized, SUPPORT_TERMS):
        topic = "support"

    if any(token in normalized.split() for token in QUESTION_TERMS) or "?" in text:
        intent = "question"

    if _contains_any(normalized, HIGH_RISK_TERMS):
        sentiment = "negative"
        severity = "high"
        urgency = "high"
        needs_human_review = True
        review_reason = "high_risk_feedback"

    return {
        "sentiment": sentiment,
        "intent": intent,
        "topic": topic,
        "severity": severity,
        "urgency": urgency,
        "needs_human_review": needs_human_review,
        "review_reason": review_reason,
        "confidence": 0.55,
        "input_type": "text",
        "emoji_detected": False,
        "human_readable": text,
    }


def _sanitize_llm_output(parsed: dict, fallback_text: str) -> dict:
    sanitized = dict(parsed or {})
    if sanitized.get("sentiment") not in VALID_SENTIMENTS:
        sanitized["sentiment"] = "neutral"
    if sanitized.get("intent") not in VALID_INTENTS:
        sanitized["intent"] = "question" if "?" in fallback_text else "complaint"
    if sanitized.get("topic") not in VALID_TOPICS:
        sanitized["topic"] = "general"
    if sanitized.get("severity") not in VALID_LEVELS:
        sanitized["severity"] = "low"
    if sanitized.get("urgency") not in VALID_LEVELS:
        sanitized["urgency"] = "low"

    try:
        sanitized["confidence"] = float(sanitized.get("confidence", 0.5))
    except (TypeError, ValueError):
        sanitized["confidence"] = 0.5

    sanitized["needs_human_review"] = bool(sanitized.get("needs_human_review", False))
    sanitized["review_reason"] = str(sanitized.get("review_reason", "") or "")
    sanitized["input_type"] = sanitized.get("input_type", "text")
    sanitized["emoji_detected"] = bool(sanitized.get("emoji_detected", False))
    sanitized["human_readable"] = sanitized.get("human_readable", fallback_text)
    return sanitized


def _apply_minimal_overrides(model_output: dict, fallback: dict, raw_text: str) -> dict:
    merged = dict(model_output)
    normalized = _normalize_text(raw_text)

    if _contains_any(normalized, HIGH_RISK_TERMS):
        merged["sentiment"] = "negative"
        merged["severity"] = "high"
        merged["urgency"] = "high"
        merged["needs_human_review"] = True
        if not merged.get("review_reason"):
            merged["review_reason"] = "high_risk_feedback"

    if merged["confidence"] < 0.45:
        merged["needs_human_review"] = True
        if not merged.get("review_reason"):
            merged["review_reason"] = "low_confidence_classification"

    if merged.get("topic") == "general" and fallback.get("topic") != "general":
        merged["topic"] = fallback["topic"]

    if not merged.get("human_readable"):
        merged["human_readable"] = fallback["human_readable"]

    return merged


def understand(cleaned_feedback: str) -> dict:
    fallback = _fallback_understanding(cleaned_feedback)
    chain = prompt | llm
    result = chain.invoke({"cleaned_feedback": cleaned_feedback})
    text = result.content.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        parsed = _sanitize_llm_output(json.loads(text), cleaned_feedback)
        return _apply_minimal_overrides(parsed, fallback, cleaned_feedback)
    except json.JSONDecodeError:
        return fallback
