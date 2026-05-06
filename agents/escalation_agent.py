"""
agents/escalation_agent.py  (replace your existing escalation_agent.py)
─────────────────────────────────────────────────────────────────────────────
v5 changes (Approach 2 — authenticated users):
  • escalate() accepts customer_name, customer_email, customer_phone
  • Passes customer details to crm_agent (Contact linked to ticket)
  • Passes customer details to email_agent (auto escalation email sent)
  • All external calls wrapped in try/except — nothing crashes the pipeline
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
  • Use domain-appropriate language for {sector}."""

_HUMAN = """Customer feedback: {feedback}
Intent: {intent} | Topic: {topic} | Severity: {severity} | Urgency: {urgency}
Escalation reason: {reason}
Edge cases: {edge_cases}
Summary: {human_readable}

Write the handoff note:"""


def escalate(
    feedback: str,
    understanding: dict,
    sector: str | None = None,
    customer_name: str = "Unknown",
    customer_email: str = "",
    customer_phone: str = "",
) -> dict:
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
        "feedback":       feedback,
        "intent":         understanding.get("intent", "unknown"),
        "topic":          understanding.get("topic", "general"),
        "severity":       understanding.get("severity", "unknown"),
        "urgency":        understanding.get("urgency", "unknown"),
        "reason":         reason,
        "edge_cases":     ", ".join(edge_cases) if edge_cases else "none",
        "human_readable": understanding.get("human_readable", feedback),
    }).content.strip()

    case = {
        "status":          "escalated",
        "feedback":        feedback,
        "handoff_note":    handoff_note,
        "reason":          reason,
        "intent":          understanding.get("intent"),
        "sentiment":       understanding.get("sentiment"),
        "topic":           understanding.get("topic", "general"),
        "severity":        understanding.get("severity"),
        "urgency":         understanding.get("urgency"),
        "confidence":      understanding.get("confidence"),
        "edge_cases":      edge_cases,
        "sector":          effective_sector,
        "assigned_to":     "human_support_team",
        "customer_name":   customer_name,
        "customer_email":  customer_email,
    }

    print(f"[ESCALATION] {handoff_note}")

    # ── CRM: create ticket + contact in HubSpot ───────────────────────────
    try:
        from agents.crm_agent import create_ticket
        crm_result = create_ticket(
            feedback       = feedback,
            analysis       = {**understanding, "sector": effective_sector},
            status         = "escalated",
            handoff_note   = handoff_note,
            customer_name  = customer_name,
            customer_email = customer_email,
            customer_phone = customer_phone,
        )
        case["crm_ticket_id"]  = crm_result.get("ticket_id")
        case["crm_ticket_url"] = crm_result.get("ticket_url")
        case["crm_status"]     = crm_result.get("crm_status")
    except Exception as exc:
        print(f"[CRM] Failed (non-fatal): {exc}")
        case["crm_ticket_id"]  = None
        case["crm_ticket_url"] = None
        case["crm_status"]     = "error"

    # ── Email: notify customer that their case was escalated ──────────────
    try:
        from agents.email_agent import send_escalation_email
        if customer_email:
            send_escalation_email(
                customer_email = customer_email,
                customer_name  = customer_name,
                ticket_id      = case.get("crm_ticket_id") or "N/A",
                feedback       = feedback,
                sector         = effective_sector,
            )
    except Exception as exc:
        print(f"[EMAIL] Failed (non-fatal): {exc}")

    return case