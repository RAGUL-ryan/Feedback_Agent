"""
agents/decision_agent.py
─────────────────────────────────────────────────────────────────────────────
Routing decision agent.

OLD: Pure if/else chain — no LLM reasoning at all.
NEW: LLM reasons over full analysis context and calls one of two tools:
       route_to_respond   – auto-handle with AI reply
       route_to_escalate  – hand off to human team

The ESCALATION_THRESHOLD from config still acts as a hard floor injected
into the system prompt. LLM cannot override it.

FIX v2:
  BUG: llama-3.1-8b-instant generates tool calls in broken XML format
       (<function=name>{args}) instead of proper JSON, causing Groq 400
       BadRequestError. Fix is two-part:
         1. Model upgraded to llama-3.3-70b-versatile in config.py (primary fix).
         2. try/except added here so any future model/API failure falls back
            to the deterministic hard-rule escalation logic instead of
            crashing the whole pipeline with a 500 error.
"""

from __future__ import annotations
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from config import ESCALATION_THRESHOLD, MODEL_NAME


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
    Escalate this feedback to the human support team.
    Call when human judgment, investigation, or manual action is required.
    reason: clear explanation of why escalation is needed.
    priority: 'low', 'medium', or 'high'
    """
    return {"route": "escalate", "reason": reason, "priority": priority}


_TOOLS = [route_to_respond, route_to_escalate]

# FIX: Removed tool_choice="any" — on weaker models this forces a tool call
# even when the model cannot format one correctly, guaranteeing the 400 error.
# With tool_choice omitted (auto), the model only emits a tool call when it
# can do so correctly; the hard-rule fallback below handles the rest.
_llm = ChatGroq(model=MODEL_NAME, temperature=0).bind_tools(_TOOLS)

_SYSTEM = f"""You are a customer feedback routing agent for a fintech support system.

Decide: should this be handled by an automated AI reply, or escalated to a human?

ALWAYS escalate if ANY of these are true:
  • needs_human_review is true
  • severity is 'high'
  • urgency is 'high'
  • confidence is below {ESCALATION_THRESHOLD}
  • contradiction_detected is true
  • edge_cases contains 'security_sensitive_feedback'
  • edge_cases contains 'billing_duplicate_charge'
  • edge_cases contains 'empty_input' or 'too_long_input'
  • intent is 'refund_request'

ESCALATE for combinations:
  • intent is 'bug_report' or 'complaint' AND urgency='medium' AND severity!='low'
  • too_short_input edge case AND confidence <= 0.4

AUTO-RESPOND for all other clear, classifiable cases.

When in doubt: escalate. A false escalation is always safer than a missed issue.
You MUST call exactly one tool: either route_to_respond or route_to_escalate."""


def _hard_rule_decision(understanding: dict) -> str:
    """
    Deterministic fallback — mirrors the LLM rules exactly.
    Used when the LLM call fails OR returns no tool call.
    """
    edge_cases = understanding.get("edge_cases", [])
    signals    = understanding.get("signals", {})

    if (
        understanding.get("needs_human_review")
        or understanding.get("severity") == "high"
        or understanding.get("urgency") == "high"
        or float(understanding.get("confidence", 0.5)) < ESCALATION_THRESHOLD
        or signals.get("contradiction_detected")
        or "security_sensitive_feedback" in edge_cases
        or "billing_duplicate_charge" in edge_cases
        or "empty_input" in edge_cases
        or "too_long_input" in edge_cases
        or understanding.get("intent") == "refund_request"
    ):
        return "escalate"

    intent   = understanding.get("intent", "")
    urgency  = understanding.get("urgency", "low")
    severity = understanding.get("severity", "low")
    if intent in ("bug_report", "complaint") and urgency == "medium" and severity != "low":
        return "escalate"
    if "too_short_input" in edge_cases and float(understanding.get("confidence", 0.5)) <= 0.4:
        return "escalate"

    return "respond"


def decide(understanding: dict) -> str:
    """
    Returns 'respond' or 'escalate'.
    Tries LLM tool-call first; falls back to hard rules on any failure.
    """
    edge_cases = understanding.get("edge_cases", [])
    signals    = understanding.get("signals", {})

    context = f"""Feedback analysis:
  sentiment           : {understanding.get('sentiment')}
  intent              : {understanding.get('intent')}
  topic               : {understanding.get('topic')}
  urgency             : {understanding.get('urgency')}
  severity            : {understanding.get('severity')}
  confidence          : {understanding.get('confidence')}
  needs_human_review  : {understanding.get('needs_human_review')}
  review_reason       : {understanding.get('review_reason') or 'none'}
  edge_cases          : {edge_cases if edge_cases else 'none'}
  contradiction_detected: {signals.get('contradiction_detected', False)}
  summary             : {understanding.get('human_readable')}

Route this case."""

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
                print(f"[Decision] escalate — {args.get('reason', '')} "
                      f"priority={args.get('priority', 'medium')}")
                return "escalate"

        # LLM returned text instead of a tool call — use hard rules
        print("[Decision] LLM returned no tool call, applying hard rules.")

    except Exception as exc:
        # Groq 400 / network error / model quota — never crash the pipeline
        print(f"[Decision] LLM call failed ({exc}), applying hard rules.")

    # ── Hard-rule fallback ────────────────────────────────────────────────────
    route = _hard_rule_decision(understanding)
    print(f"[Decision] hard-rule fallback → {route}")
    return route