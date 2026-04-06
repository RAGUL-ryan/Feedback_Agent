"""
agents/escalation_agent.py
─────────────────────────────────────────────────────────────────────────────
Escalation packaging agent.

v3 changes:
  • Sector-aware system prompt — handoff note uses domain-appropriate language
    (e.g. "patient" for healthcare, "order" for ecommerce, "transaction" for fintech)
  • escalate() accepts optional `sector` param; falls back to config.SECTOR
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from config import MODEL_NAME, SECTOR

_llm = ChatGroq(model=MODEL_NAME, temperature=0)

_SECTOR_ROLE = {
    "fintech":    "fintech customer support team",
    "education":  "education support team",
    "healthcare": "patient care and billing support team",
    "food":       "food delivery support team",
    "ecommerce":  "ecommerce customer support team",
}

_SYSTEM_TEMPLATE = """You are a customer support escalation coordinator for a {sector_role}.

Write a concise escalation handoff note for the human support agent.
Include:
  1. What the customer reported (in plain language relevant to {sector})
  2. Why this is being escalated
  3. What the agent should do first

Rules:
  • Under 60 words.
  • Clear, professional English sentences — no bullet points.
  • Do not mention AI confidence scores or internal system labels.
  • Use domain-appropriate language for {sector} (e.g. "patient" not "customer" for healthcare)."""

_HUMAN = """Customer feedback: {feedback}
Intent: {intent} | Topic: {topic} | Severity: {severity} | Urgency: {urgency}
Escalation reason: {reason}
Edge cases: {edge_cases}
Summary: {human_readable}

Write the handoff note:"""


def escalate(feedback: str, understanding: dict, sector: str | None = None) -> dict:
    effective_sector = (sector or understanding.get("sector") or SECTOR or "fintech").lower()
    sector_role = _SECTOR_ROLE.get(effective_sector, "customer support team")

    edge_cases = understanding.get("edge_cases", [])
    reason = (
        understanding.get("review_reason")
        or f"urgency={understanding.get('urgency')} severity={understanding.get('severity')} confidence={understanding.get('confidence')}"
    )

    system_msg = _SYSTEM_TEMPLATE.format(sector=effective_sector, sector_role=sector_role)
    prompt = ChatPromptTemplate.from_messages([("system", system_msg), ("human", _HUMAN)])

    handoff_note = (prompt | _llm).invoke({
        "feedback": feedback,
        "intent": understanding.get("intent", "unknown"),
        "topic": understanding.get("topic", "general"),
        "severity": understanding.get("severity", "unknown"),
        "urgency": understanding.get("urgency", "unknown"),
        "reason": reason,
        "edge_cases": ", ".join(edge_cases) if edge_cases else "none",
        "human_readable": understanding.get("human_readable", feedback),
    }).content.strip()

    case = {
        "status": "escalated",
        "feedback": feedback,
        "handoff_note": handoff_note,
        "reason": reason,
        "intent": understanding.get("intent"),
        "sentiment": understanding.get("sentiment"),
        "topic": understanding.get("topic", "general"),
        "severity": understanding.get("severity"),
        "urgency": understanding.get("urgency"),
        "confidence": understanding.get("confidence"),
        "edge_cases": edge_cases,
        "sector": effective_sector,
        "assigned_to": "human_support_team",
    }

    print(f"[ESCALATION] {handoff_note}")
    # TODO: plug in tools/notification.py → Slack / Zendesk / Jira
    return case