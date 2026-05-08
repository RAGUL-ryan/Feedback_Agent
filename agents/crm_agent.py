"""
agents/crm_agent.py
─────────────────────────────────────────────────────────────────────────────
CRM integration agent — HubSpot (FREE tier).

v3 changes (unchanged from v2 functionally):
  This file has NO routing logic. It does exactly one thing: create a ticket
  and link a contact. Whether to call this at all is decided upstream by
  decision_agent (via escalation_agent). Nothing here is hardcoded about
  which feedback types trigger a ticket.

HubSpot free tier includes:
  ✓ Unlimited tickets + contacts
  ✓ Contact linked to ticket (full customer profile visible to agent)
  ✓ No credit card required
"""

from __future__ import annotations

import os
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

HUBSPOT_TOKEN = os.getenv("HUBSPOT_ACCESS_TOKEN")
HUBSPOT_BASE  = "https://api.hubapi.com"

_STAGE_MAP = {
    "escalated": "1",   # New — needs human action
    "replied":   "4",   # Closed — AI handled it
}


def _is_configured() -> bool:
    return bool(HUBSPOT_TOKEN)


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {HUBSPOT_TOKEN}",
        "Content-Type":  "application/json",
    }


def _map_priority(urgency: str, severity: str, status: str) -> str:
    if status == "escalated":
        if urgency == "high" or severity == "high":
            return "HIGH"
        elif urgency == "medium" or severity == "medium":
            return "MEDIUM"
        return "LOW"
    return "LOW"


def _get_or_create_contact(
    email: str,
    name: str,
    phone: str | None,
) -> str | None:
    """
    Find existing HubSpot contact by email or create a new one.
    Returns the HubSpot contact ID (string), or None on failure.
    HubSpot automatically deduplicates by email — safe to call every time.
    """
    if not email:
        return None

    first, *rest = name.strip().split(" ", 1)
    last = rest[0] if rest else ""

    try:
        resp = requests.post(
            f"{HUBSPOT_BASE}/crm/v3/objects/contacts",
            headers=_headers(),
            json={
                "properties": {
                    "email":     email,
                    "firstname": first,
                    "lastname":  last,
                    "phone":     phone or "",
                }
            },
            timeout=10,
        )

        if resp.status_code == 201:
            contact_id = resp.json().get("id")
            print(f"[CRM] Contact created → id={contact_id}")
            return contact_id

        if resp.status_code == 409:
            search = requests.post(
                f"{HUBSPOT_BASE}/crm/v3/objects/contacts/search",
                headers=_headers(),
                json={
                    "filterGroups": [{
                        "filters": [{
                            "propertyName": "email",
                            "operator":     "EQ",
                            "value":        email,
                        }]
                    }]
                },
                timeout=10,
            )
            results = search.json().get("results", [])
            if results:
                contact_id = results[0]["id"]
                print(f"[CRM] Existing contact found → id={contact_id}")
                return contact_id

    except Exception as exc:
        print(f"[CRM] Contact error (non-fatal): {exc}")

    return None


def _link_contact_to_ticket(ticket_id: str, contact_id: str):
    """Associate a HubSpot contact with a ticket using the Associations API."""
    try:
        requests.put(
            f"{HUBSPOT_BASE}/crm/v3/objects/tickets/{ticket_id}/associations/contacts/{contact_id}/ticket_to_contact",
            headers=_headers(),
            timeout=10,
        )
        print(f"[CRM] Contact {contact_id} linked to ticket {ticket_id}")
    except Exception as exc:
        print(f"[CRM] Link error (non-fatal): {exc}")


def _build_description(
    feedback: str,
    analysis: dict,
    status: str,
    handoff_note: str,
    reply: str,
    customer_name: str,
) -> str:
    intent     = analysis.get("intent", "general")
    topic      = analysis.get("topic", "general")
    urgency    = analysis.get("urgency", "low")
    severity   = analysis.get("severity", "low")
    sector     = analysis.get("sector", "general")
    sentiment  = analysis.get("sentiment", "neutral")
    confidence = analysis.get("confidence", "n/a")
    edge_cases = analysis.get("edge_cases", [])
    human_readable = analysis.get("human_readable", feedback)
    timestamp  = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    if status == "escalated":
        return (
            f"[AI ESCALATION — {sector.upper()}]\n\n"
            f"Customer: {customer_name}\n"
            f"Feedback: {feedback}\n\n"
            f"Plain English: {human_readable}\n\n"
            f"Handoff note:\n{handoff_note or 'See analysis below.'}\n\n"
            f"--- Pipeline Analysis ---\n"
            f"Intent    : {intent}\n"
            f"Topic     : {topic}\n"
            f"Urgency   : {urgency}\n"
            f"Severity  : {severity}\n"
            f"Sentiment : {sentiment}\n"
            f"Confidence: {confidence}\n"
            f"Edge cases: {', '.join(edge_cases) if edge_cases else 'none'}\n"
            f"Timestamp : {timestamp}"
        )
    return (
        f"[AI AUTO-REPLIED — {sector.upper()}]\n\n"
        f"Customer: {customer_name}\n"
        f"Feedback: {feedback}\n\n"
        f"AI reply:\n{reply or 'See logs.'}\n\n"
        f"Intent: {intent} | Sentiment: {sentiment}\n"
        f"Timestamp: {timestamp}"
    )


def create_ticket(
    feedback: str,
    analysis: dict,
    status: str,
    handoff_note: str = "",
    reply: str = "",
    customer_name: str = "Unknown",
    customer_email: str = "",
    customer_phone: str = "",
) -> dict:
    """
    Create a HubSpot ticket and link it to a Contact (the customer).
    This function is called ONLY when the decision_agent determined a ticket
    is warranted. It does not make that judgment itself.

    Args:
        feedback       : cleaned customer feedback text
        analysis       : full understanding dict from pipeline
        status         : 'escalated' | 'replied'
        handoff_note   : handoff note (escalated cases)
        reply          : AI reply text (replied cases)
        customer_name  : from JWT token
        customer_email : from JWT token
        customer_phone : from JWT token

    Returns:
        dict with crm_status, ticket_id, ticket_url
    """
    if not _is_configured():
        print("[CRM] HUBSPOT_ACCESS_TOKEN not set — skipping.")
        return {"crm_status": "skipped", "reason": "no_token"}

    intent   = analysis.get("intent", "general")
    urgency  = analysis.get("urgency", "low")
    severity = analysis.get("severity", "low")
    sector   = analysis.get("sector", "general")

    priority    = _map_priority(urgency, severity, status)
    stage       = _STAGE_MAP.get(status, "1")
    subject     = f"[{status.upper()}] {sector} | {intent} | {urgency} urgency | {customer_name}"
    description = _build_description(
        feedback, analysis, status, handoff_note, reply, customer_name
    )

    try:
        # ── Step 1: Create the ticket ─────────────────────────────────────────
        resp = requests.post(
            f"{HUBSPOT_BASE}/crm/v3/objects/tickets",
            headers=_headers(),
            json={
                "properties": {
                    "subject":            subject,
                    "content":            description,
                    "hs_ticket_priority": priority,
                    "hs_pipeline":        "0",
                    "hs_pipeline_stage":  stage,
                    "source_type":        "CHAT",
                }
            },
            timeout=10,
        )
        resp.raise_for_status()
        ticket_id  = resp.json().get("id", "unknown")
        ticket_url = f"https://app.hubspot.com/contacts/tickets/{ticket_id}"
        print(f"[CRM] ✓ Ticket created → {ticket_url}")

        # ── Step 2: Create/find Contact and link to ticket ────────────────────
        if customer_email:
            contact_id = _get_or_create_contact(
                customer_email, customer_name, customer_phone
            )
            if contact_id:
                _link_contact_to_ticket(ticket_id, contact_id)

        return {
            "crm_status": "created",
            "ticket_id":  ticket_id,
            "ticket_url": ticket_url,
            "priority":   priority,
        }

    except requests.exceptions.HTTPError as e:
        print(f"[CRM] HTTP error: {e.response.status_code} — {e.response.text}")
        return {"crm_status": "error", "reason": str(e)}
    except requests.exceptions.RequestException as e:
        print(f"[CRM] Network error: {e}")
        return {"crm_status": "error", "reason": str(e)}