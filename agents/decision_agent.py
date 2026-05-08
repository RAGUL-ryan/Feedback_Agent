"""
agents/decision_agent.py
─────────────────────────────────────────────────────────────────────────────
Routing decision agent.

v3 changes — Agent-driven CRM/email gating:
  • Added a THIRD route: escalate_with_ticket
      - escalate          → human review needed, but no formal ticket required
                            (e.g. vague complaint, low-confidence classification,
                             short ambiguous input, mild negative sentiment)
      - escalate_with_ticket → serious case that ALSO needs a CRM ticket +
                            customer email notification
                            (e.g. fraud, double-charge, refund request,
                             high-severity bug, regulatory mention)

  The LLM reasons over the full analysis context and calls ONE of three tools:
      route_to_respond         – auto-handle with AI reply
      route_to_escalate        – hand off to human, NO ticket
      route_to_escalate_ticket – hand off to human AND create CRM ticket + email

  Hard-rule fallback mirrors the same 3-way logic — used when LLM fails.

  IMPORTANT: Nothing is hardcoded about WHICH intents/topics trigger a ticket.
  The LLM decides based on the principles injected in the system prompt.
  Only the PRINCIPLES are listed — not a fixed keyword list.
"""

from __future__ import annotations
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from config import ESCALATION_THRESHOLD, MODEL_NAME


# ─────────────────────────────────────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────────────────────────────────────

@tool
def route_to_respond(reason: str) -> dict:
    """
    Route this feedback to the automated response agent.
    Call when the case is clear enough for an accurate AI-generated reply.
    reason: brief explanation of why auto-response is appropriate.
    """
    return {"route": "respond", "reason": reason}


@tool
def route_to_escalate(reason: str, priority: str) -> dict:
    """
    Escalate this feedback to the human support team. Do NOT create a CRM ticket.
    Call when human judgment is needed but the case does not require formal
    tracking — e.g. low-confidence classification, vague complaint, ambiguous input,
    minor sensitivity, or any case where a ticket would be overkill.
    reason: clear explanation of why escalation is needed.
    priority: 'low', 'medium', or 'high'
    """
    return {"route": "escalate", "reason": reason, "priority": priority}


@tool
def route_to_escalate_ticket(reason: str, priority: str, ticket_reason: str) -> dict:
    """
    Escalate this feedback to the human support team AND create a CRM ticket
    + send a customer notification email.

    Call when the case is both serious enough to escalate AND requires formal
    tracking with a ticket number and a customer-facing email. Use your judgment —
    the core question is: "Would a real support team want a formal ticket for this?"

    Principles that suggest a ticket IS warranted:
      • The customer is owed a formal response with a reference number
      • The case involves money, account integrity, or legal/regulatory language
      • The issue cannot be resolved without back-and-forth and needs tracking
      • High severity OR high urgency on a substantive (not trivial) complaint
      • The customer explicitly asked for follow-up, a callback, or a resolution
      • Safety, fraud, or data-breach concerns — even potential ones

    Principles that suggest a ticket is NOT warranted (use route_to_escalate instead):
      • The input is too short/vague to create a meaningful ticket
      • Low confidence means we don't know what the issue really is
      • The case is borderline and a ticket would confuse or alarm the customer

    reason: why escalation is needed.
    priority: 'low', 'medium', or 'high'
    ticket_reason: one sentence explaining why a formal ticket is also appropriate.
    """
    return {
        "route": "escalate_with_ticket",
        "reason": reason,
        "priority": priority,
        "ticket_reason": ticket_reason,
    }


_TOOLS = [route_to_respond, route_to_escalate, route_to_escalate_ticket]

_llm = ChatGroq(model=MODEL_NAME, temperature=0).bind_tools(_TOOLS)

_SYSTEM = f"""You are a customer feedback routing agent for a multi-sector support system.

Decide which of THREE routes is correct for this feedback:
  1. route_to_respond          – AI can handle it directly
  2. route_to_escalate         – human review needed, no formal ticket
  3. route_to_escalate_ticket  – human review needed AND a CRM ticket + email

━━━ ROUTING RULES ━━━

ALWAYS call route_to_respond when:
  • Confidence is high (≥ {ESCALATION_THRESHOLD}) AND no escalation signals exist
  • Clear question, suggestion, praise, or routine complaint with obvious resolution

ALWAYS call route_to_escalate OR route_to_escalate_ticket (never route_to_respond) if ANY of:
  • needs_human_review is true
  • severity is 'high'
  • urgency is 'high'
  • confidence is below {ESCALATION_THRESHOLD}
  • contradiction_detected is true
  • edge_cases contains 'empty_input', 'too_long_input'
  • edge_cases contains 'security_sensitive_feedback'
  • edge_cases contains 'billing_duplicate_charge'
  • intent is 'refund_request'
  • intent is 'bug_report' or 'complaint' AND urgency='medium' AND severity!='low'
  • too_short_input AND confidence ≤ 0.4

━━━ TICKET DECISION (only applies when escalating) ━━━

Once you decide to escalate, ask: "Does this case warrant a formal CRM ticket
and a customer notification email?"

Call route_to_escalate_ticket (ticket YES) when the case involves:
  • Any financial concern — overcharge, double payment, missing refund, wrong deduction
  • Any security/fraud/account-integrity concern — even a suspected one
  • A refund request or explicit request for follow-up
  • High severity on a substantive issue (not just ambiguous/short input)
  • Legal language, regulatory mention, or safety concern
  • A specific, articulable problem the customer deserves a reference number for

Call route_to_escalate (ticket NO) when:
  • The input is too vague, short, or ambiguous to create a meaningful ticket
  • Low confidence is the primary escalation trigger — we don't know enough yet
  • It is a borderline case with no financial, security, or regulatory angle

When in doubt: escalate. If unsure about ticket: no ticket is safer than a noisy one.
You MUST call exactly one tool."""


# ─────────────────────────────────────────────────────────────────────────────
# Hard-rule fallback — deterministic mirror of the LLM rules
# ─────────────────────────────────────────────────────────────────────────────

def _hard_rule_decision(understanding: dict) -> str:
    """
    Returns 'respond', 'escalate', or 'escalate_with_ticket'.
    Used when the LLM call fails or returns no tool call.
    Mirrors the LLM system prompt logic exactly.
    """
    edge_cases = understanding.get("edge_cases", [])
    signals    = understanding.get("signals", {})
    intent     = understanding.get("intent", "")
    urgency    = understanding.get("urgency", "low")
    severity   = understanding.get("severity", "low")
    confidence = float(understanding.get("confidence", 0.5))

    # ── Is escalation needed at all? ─────────────────────────────────────────
    needs_escalation = (
        understanding.get("needs_human_review")
        or severity == "high"
        or urgency == "high"
        or confidence < ESCALATION_THRESHOLD
        or signals.get("contradiction_detected")
        or "security_sensitive_feedback" in edge_cases
        or "billing_duplicate_charge" in edge_cases
        or "empty_input" in edge_cases
        or "too_long_input" in edge_cases
        or intent == "refund_request"
        or (intent in ("bug_report", "complaint") and urgency == "medium" and severity != "low")
        or ("too_short_input" in edge_cases and confidence <= 0.4)
    )

    if not needs_escalation:
        return "respond"

    # ── Does this escalated case warrant a ticket? ────────────────────────────
    ticket_signals = (
        "security_sensitive_feedback" in edge_cases
        or "billing_duplicate_charge" in edge_cases
        or intent == "refund_request"
        or (severity == "high" and intent not in ("general",) and confidence > 0.4)
        or (urgency == "high" and len(understanding.get("human_readable", "").split()) > 4)
    )

    # Vague / low-confidence cases: escalate but no ticket
    vague_only = (
        confidence < ESCALATION_THRESHOLD
        and not ticket_signals
        and "too_short_input" in edge_cases
        or ("empty_input" in edge_cases)
        or ("too_long_input" in edge_cases and not ticket_signals)
    )

    if ticket_signals and not vague_only:
        return "escalate_with_ticket"

    return "escalate"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def decide(understanding: dict) -> str:
    """
    Returns 'respond', 'escalate', or 'escalate_with_ticket'.
    Tries LLM tool-call first; falls back to hard rules on any failure.
    """
    edge_cases = understanding.get("edge_cases", [])
    signals    = understanding.get("signals", {})

    context = f"""Feedback analysis:
  sentiment             : {understanding.get('sentiment')}
  intent                : {understanding.get('intent')}
  topic                 : {understanding.get('topic')}
  urgency               : {understanding.get('urgency')}
  severity              : {understanding.get('severity')}
  confidence            : {understanding.get('confidence')}
  needs_human_review    : {understanding.get('needs_human_review')}
  review_reason         : {understanding.get('review_reason') or 'none'}
  edge_cases            : {edge_cases if edge_cases else 'none'}
  contradiction_detected: {signals.get('contradiction_detected', False)}
  summary               : {understanding.get('human_readable')}

Route this case. Remember: choose route_to_escalate_ticket only when a formal
CRM ticket and customer email are genuinely warranted — not just because escalation
is needed."""

    # ── Try LLM tool-call ─────────────────────────────────────────────────────
    try:
        response: AIMessage = _llm.invoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=context),
        ])

        if response.tool_calls:
            tc   = response.tool_calls[0]
            name = tc["name"]
            args = tc["args"]

            if name == "route_to_respond":
                print(f"[Decision] respond — {args.get('reason', '')}")
                return "respond"

            if name == "route_to_escalate":
                print(f"[Decision] escalate (no ticket) — {args.get('reason', '')} "
                      f"priority={args.get('priority', 'medium')}")
                return "escalate"

            if name == "route_to_escalate_ticket":
                print(f"[Decision] escalate + ticket — {args.get('reason', '')} "
                      f"| ticket: {args.get('ticket_reason', '')} "
                      f"priority={args.get('priority', 'medium')}")
                return "escalate_with_ticket"

        print("[Decision] LLM returned no tool call, applying hard rules.")

    except Exception as exc:
        print(f"[Decision] LLM call failed ({exc}), applying hard rules.")

    # ── Hard-rule fallback ────────────────────────────────────────────────────
    route = _hard_rule_decision(understanding)
    print(f"[Decision] hard-rule fallback → {route}")
    return route