"""
agents/escalation_agent.py
─────────────────────────────────────────────────────────────────────────────
v7 changes — Agent-driven CRM + email gating:

  The `escalate()` function now accepts a `create_ticket` boolean flag
  that is set by the decision_agent (via main.py).

  When create_ticket=True  → crm_agent creates a HubSpot ticket and
                              email_agent sends a customer notification.
  When create_ticket=False → inline acknowledgment only (previous v6 behaviour).

  Nothing about *which* cases get tickets is hardcoded here.
  The decision_agent owns that logic entirely.
"""

from langchain_core.prompts import ChatPromptTemplate
#from langchain_groq import ChatGroq
from config import MODEL_NAME, SECTOR

#_llm = ChatGroq(model=MODEL_NAME, temperature=0)

from langchain_openai import ChatOpenAI
_llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
_SECTOR_ROLE = {
    "fintech":    "fintech customer support team",
    "education":  "education support team",
    "healthcare": "patient care and billing support team",
    "food":       "food delivery support team",
    "ecommerce":  "ecommerce customer support team",
}

# ── Handoff note (internal — not shown to customer) ──────────────────────────
_HANDOFF_SYSTEM = """You are a customer support escalation coordinator for a {sector_role}.

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

_HANDOFF_HUMAN = """Customer feedback: {feedback}
Intent: {intent} | Topic: {topic} | Severity: {severity} | Urgency: {urgency}
Escalation reason: {reason}
Edge cases: {edge_cases}
Summary: {human_readable}

Write the handoff note:"""

# ── Customer-facing acknowledgment (shown in UI) ──────────────────────────────
_ACK_SYSTEM = """You are a warm, empathetic customer support representative for a {sector_role}.

Write a personal acknowledgment reply shown directly to the customer in the app UI —
exactly like how a host replies to a guest review on Airbnb.

The customer left negative or complex feedback. Your reply must:
  1. Address the customer by first name (use {customer_name})
  2. Acknowledge their specific concern genuinely — do not be dismissive
  3. Take appropriate ownership (even if the issue has an explanation)
  4. Tell them what happens next in one clear sentence
  5. End on a warm, constructive note

Rules:
  • 3–4 sentences maximum.
  • Natural, human tone — not robotic corporate language.
  • Do NOT use vague phrases like "our team will reach out". Be specific about next step.
  • If a ticket was created, you MAY reference that a case reference has been created
    for tracking — but do NOT invent a ticket number; leave that to the email.
  • If no ticket was created, do NOT mention emails, tickets, or reference numbers.
  • Do NOT reveal internal system details (confidence scores, edge case flags).
  • Use domain-appropriate language for {sector}."""

_ACK_HUMAN = """Customer feedback: {feedback}
What they mean: {human_readable}
Intent: {intent} | Topic: {topic} | Severity: {severity} | Urgency: {urgency}
Customer first name: {customer_name}
Formal ticket created: {ticket_created}

Write the acknowledgment reply shown in the UI:"""


def escalate(
    feedback: str,
    understanding: dict,
    create_ticket: bool = False,
    sector: str | None = None,
    customer_name: str = "there",
    customer_email: str = "",
    customer_phone: str = "",
) -> dict:
    """
    Generate escalation outputs.

    Args:
        feedback       : cleaned customer feedback text
        understanding  : full analysis dict from the pipeline
        create_ticket  : if True, create CRM ticket + send email notification.
                         Set by decision_agent — NOT hardcoded here.
        sector         : optional sector override
        customer_name  : from JWT token
        customer_email : from JWT token
        customer_phone : from JWT token

    Returns:
        dict with status, acknowledgment, handoff_note, crm/email results, etc.
    """
    effective_sector = (sector or understanding.get("sector") or SECTOR or "fintech").lower()
    sector_role      = _SECTOR_ROLE.get(effective_sector, "customer support team")

    edge_cases = understanding.get("edge_cases", [])
    reason = (
        understanding.get("review_reason")
        or f"urgency={understanding.get('urgency')} severity={understanding.get('severity')} "
           f"confidence={understanding.get('confidence')}"
    )

    first_name = (customer_name or "there").split()[0]

    # ── Generate internal handoff note ────────────────────────────────────────
    handoff_system = _HANDOFF_SYSTEM.format(sector=effective_sector, sector_role=sector_role)
    handoff_prompt = ChatPromptTemplate.from_messages([
        ("system", handoff_system),
        ("human", _HANDOFF_HUMAN),
    ])
    handoff_note = (handoff_prompt | _llm).invoke({
        "feedback":       feedback,
        "intent":         understanding.get("intent", "unknown"),
        "topic":          understanding.get("topic", "general"),
        "severity":       understanding.get("severity", "unknown"),
        "urgency":        understanding.get("urgency", "unknown"),
        "reason":         reason,
        "edge_cases":     ", ".join(edge_cases) if edge_cases else "none",
        "human_readable": understanding.get("human_readable", feedback),
    }).content.strip()

    # ── Generate customer-facing acknowledgment ───────────────────────────────
    ack_system = _ACK_SYSTEM.format(
        sector=effective_sector,
        sector_role=sector_role,
        customer_name=first_name,
    )
    ack_prompt = ChatPromptTemplate.from_messages([
        ("system", ack_system),
        ("human", _ACK_HUMAN),
    ])
    acknowledgment = (ack_prompt | _llm).invoke({
        "feedback":       feedback,
        "human_readable": understanding.get("human_readable", feedback),
        "intent":         understanding.get("intent", "unknown"),
        "topic":          understanding.get("topic", "general"),
        "severity":       understanding.get("severity", "unknown"),
        "urgency":        understanding.get("urgency", "unknown"),
        "customer_name":  first_name,
        "ticket_created": "yes" if create_ticket else "no",
    }).content.strip()

    print(f"[ESCALATION] Handoff note: {handoff_note}")
    print(f"[ESCALATION] Acknowledgment shown to customer: {acknowledgment}")
    print(f"[ESCALATION] CRM ticket requested: {create_ticket}")

    crm_result   = {"crm_status": "skipped", "reason": "not_required_by_agent"}
    email_result = {"email_status": "skipped", "reason": "not_required_by_agent"}

    # ── Conditionally create CRM ticket + send email ──────────────────────────
    if create_ticket:
        try:
            from agents.crm_agent import create_ticket as crm_create_ticket
            crm_result = crm_create_ticket(
                feedback       = feedback,
                analysis       = understanding,
                status         = "escalated",
                handoff_note   = handoff_note,
                reply          = "",
                customer_name  = customer_name,
                customer_email = customer_email,
                customer_phone = customer_phone,
            )
            print(f"[ESCALATION] CRM result: {crm_result}")
        except Exception as exc:
            print(f"[ESCALATION] CRM creation failed (non-fatal): {exc}")
            crm_result = {"crm_status": "error", "reason": str(exc)}

        if customer_email:
            try:
                from agents.email_agent import send_escalation_email
                ticket_id = crm_result.get("ticket_id", "N/A")
                sent = send_escalation_email(
                    customer_email = customer_email,
                    customer_name  = customer_name,
                    ticket_id      = ticket_id,
                    feedback       = feedback,
                    sector         = effective_sector,
                )
                email_result = {"email_status": "sent" if sent else "failed"}
                print(f"[ESCALATION] Email result: {email_result}")
            except Exception as exc:
                print(f"[ESCALATION] Email send failed (non-fatal): {exc}")
                email_result = {"email_status": "error", "reason": str(exc)}

    return {
        "status":         "escalated",
        "feedback":       feedback,
        "handoff_note":   handoff_note,
        "acknowledgment": acknowledgment,
        "reason":         reason,
        "intent":         understanding.get("intent"),
        "sentiment":      understanding.get("sentiment"),
        "topic":          understanding.get("topic", "general"),
        "severity":       understanding.get("severity"),
        "urgency":        understanding.get("urgency"),
        "confidence":     understanding.get("confidence"),
        "edge_cases":     edge_cases,
        "sector":         effective_sector,
        "assigned_to":    "human_support_team",
        "customer_name":  customer_name,
        "customer_email": customer_email,
        "ticket_created": create_ticket,
        "crm":            crm_result,
        "email":          email_result,
    }