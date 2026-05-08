"""
agents/decision_agent.py
─────────────────────────────────────────────────────────────────────────────
Production-grade routing decision agent.

ARCHITECTURE — Two focused LLM calls, zero hardcoded string matching:

  Call 1 — routing decision (tool-calling)
    Answers: "respond / escalate / escalate_with_ticket?"
    Reads the full analysis context including the actual customer text.
    Language-agnostic — the LLM handles all understanding.

  Call 2 — ticket assessment (focused JSON call)
    Runs in two situations:
      a) Call 1 returned "escalate_with_ticket" → Call 2 independently confirms.
      b) Call 1 failed or returned no tool call → Call 2 determines ticket need
         while hard rules handle the escalate/respond split.
    Asks one focused question:
      "Is there an OPEN, UNRESOLVED action the support team must take right now?"
    Returns: { "ticket_needed": bool, "reason": str }
    Cheap call — small prompt, binary output, fast.

  Hard-rule fallback — escalation decision ONLY
    Used only when Call 1 fails entirely (network down, quota, model error).
    Operates on structured pipeline signals from understanding_agent
    (severity, urgency, confidence, edge_case flags) — ZERO text matching.
    These signals are safe because understanding_agent already did the
    language work upstream in the pipeline.
    The ticket sub-decision is NOT attempted in the fallback — defaults to
    "escalate" (no ticket) if Call 2 also fails. A missed ticket is always
    safer than a noisy one at production scale.

WHY NO STRING MATCHING:
  Millions of users write in their own language, dialect, tone, and style.
  String matching on words like "came after", "already received", "not processed"
  will silently fail on Tamil, Tanglish, sarcasm, broken English, and regional
  phrasing — and you will never know it failed. Only an LLM can reliably
  understand whether a situation is open or resolved across all those variations.
"""

from __future__ import annotations
import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
#from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from config import ESCALATION_THRESHOLD, MODEL_NAME


# ─────────────────────────────────────────────────────────────────────────────
# LLM instances
# Two separate instances — different purposes, same model.
# ─────────────────────────────────────────────────────────────────────────────

# Call 1: routing decision — tool-calling, deterministic
_llm_router = ChatOpenAI(model=MODEL_NAME, temperature=0)

# Call 2: ticket assessment — plain JSON output, focused, fast
_llm_ticket = ChatOpenAI(model=MODEL_NAME, temperature=0)



# ─────────────────────────────────────────────────────────────────────────────
# Call 1 — Routing tools
# ─────────────────────────────────────────────────────────────────────────────

@tool
def route_to_respond(reason: str) -> dict:
    """
    Route this feedback to the automated response agent.
    Call when the AI can give a complete, accurate, helpful answer right now.
    reason: brief explanation of why auto-response is appropriate.
    """
    return {"route": "respond", "reason": reason}


@tool
def route_to_escalate(reason: str, priority: str) -> dict:
    """
    Escalate to the human support team. Do NOT create a CRM ticket.
    Call when human judgment is needed but no formal tracking is required —
    e.g. vague complaint, low-confidence case, ambiguous input, past frustration
    with nothing currently open, or any escalation where a ticket is overkill.
    reason: clear explanation of why escalation is needed.
    priority: 'low', 'medium', or 'high'
    """
    return {"route": "escalate", "reason": reason, "priority": priority}


@tool
def route_to_escalate_ticket(reason: str, priority: str, ticket_reason: str) -> dict:
    """
    Escalate to human support AND create a CRM ticket + customer notification email.

    The single deciding question:
      "Is there something the support team must DO right now for this specific
       customer, and does this customer deserve a reference number to track it?"

    Call this ONLY when the situation is CURRENTLY OPEN AND UNRESOLVED:
      • Money the customer is still owed — not yet refunded or returned
      • Account the customer still cannot access
      • Active fraud, security, or data-breach concern not yet addressed
      • Safety or health concern not yet resolved
      • Explicit customer request for follow-up on something still unresolved
      • Legal or regulatory concern requiring documented handling

    Do NOT call this when:
      • The customer is venting about something that already happened and is over
      • The issue was resolved — even if slowly or frustratingly
      • Feedback is too vague to create a meaningful ticket
      • Low confidence is the main escalation reason

    reason: why escalation is needed.
    priority: 'low', 'medium', or 'high'
    ticket_reason: one sentence — what specific open action must the support team take?
    """
    return {
        "route": "escalate_with_ticket",
        "reason": reason,
        "priority": priority,
        "ticket_reason": ticket_reason,
    }


_ROUTING_TOOLS = [route_to_respond, route_to_escalate, route_to_escalate_ticket]
_llm_router_with_tools = _llm_router.bind_tools(_ROUTING_TOOLS)


# ─────────────────────────────────────────────────────────────────────────────
# Call 1 — System prompt
# ─────────────────────────────────────────────────────────────────────────────

_ROUTING_SYSTEM = f"""You are a senior customer support routing agent for a global platform
serving millions of customers — across languages, dialects, tones, and writing styles.

Your job: read what the customer actually said and meant, then call exactly ONE routing tool.

━━━ STEP 1: ESCALATE OR RESPOND? ━━━

Call route_to_respond when ALL are true:
  • The AI can give a complete, accurate, helpful answer right now
  • No active unresolved issue requiring human action
  • No open safety, fraud, or financial concern
  • Confidence is acceptable (≥ {ESCALATION_THRESHOLD}) and classification is clear

Always escalate (route_to_escalate or route_to_escalate_ticket) when ANY signal present:
  • needs_human_review = true
  • severity = 'high' or urgency = 'high'
  • confidence < {ESCALATION_THRESHOLD}
  • contradiction_detected = true
  • edge_cases include: security_sensitive_feedback, billing_duplicate_charge,
    empty_input, too_long_input
  • intent is bug_report or complaint with urgency=medium AND severity≠low
  • too_short_input with confidence ≤ 0.4
  • Any active unresolved financial, account, or safety situation

━━━ STEP 2: TICKET OR NO TICKET? ━━━

Once you decide to escalate, ask:
  "Is there something the support team must DO right now for this customer?"

The answer comes from the ACTUAL SITUATION — not the intent label.
The same intent label can mean completely different things:

  "I asked for a refund and it still hasn't arrived"
  → open issue, money still owed → ticket warranted

  "I got the refund but it took 10 days instead of 2"
  → resolved, customer venting about the delay → no ticket

  "Love how '2 days delivery' magically became a week" (sarcasm)
  → past frustration, nothing currently open → no ticket

  "My money was debited twice and hasn't come back"
  → active open financial issue → ticket warranted

  "Account still locked, can't log in for 3 days"
  → open, blocking the customer right now → ticket warranted

Intent labels, urgency scores, and severity scores are SIGNALS to help you reason.
The customer's actual described situation is the ground truth.

You MUST call exactly one tool. When unsure about ticket: default to route_to_escalate."""


# ─────────────────────────────────────────────────────────────────────────────
# Call 2 — Ticket assessment
# Focused, fast, language-agnostic binary judgment.
# Runs independently of Call 1's routing decision.
# ─────────────────────────────────────────────────────────────────────────────

_TICKET_ASSESSMENT_SYSTEM = """You are a customer support triage specialist for a global platform.

You will receive a customer's feedback and a plain-English summary of what they mean.

Your ONLY job: decide whether this case requires a formal CRM ticket and a customer
notification email to be created right now.

The single question to answer:
  "Does the support team need to take a specific action for this customer RIGHT NOW,
   and does this customer deserve a reference number to track the outcome?"

Answer YES (ticket_needed: true) when the situation is CURRENTLY OPEN:
  • Money is missing and has NOT been returned yet
  • Account is inaccessible and has NOT been restored yet
  • Fraud or security breach is active and unaddressed
  • Safety or health concern has not been resolved
  • Customer explicitly asked for follow-up on something still unresolved
  • Legal or regulatory concern requiring documented tracking

Answer NO (ticket_needed: false) when:
  • The issue already happened and is over — even if it was bad or slow
  • The customer is expressing frustration or venting about a past experience
  • Sarcasm or irony about something that has already concluded
  • Feedback is too vague to create a meaningful ticket
  • Escalation is needed only because of low classification confidence

This platform serves millions of customers globally. They write in every language,
dialect, and style. Judge based on the MEANING and SITUATION — not the specific
words or language used.

Respond ONLY with valid JSON. No markdown, no explanation outside the JSON:
{
  "ticket_needed": true or false,
  "reason": "one sentence explaining your decision"
}"""


def _assess_ticket_need(
    original_feedback: str,
    human_readable: str,
    understanding: dict,
) -> bool:
    """
    Focused second LLM call: is a CRM ticket warranted for this case?

    Returns True if a ticket is needed, False otherwise.
    On any failure: returns False — the safe default at production scale.
    """
    context = f"""Customer's actual feedback:
"{original_feedback}"

Plain-English summary of what they mean:
"{human_readable}"

Supporting signals (use to inform your judgment — not as the final answer):
  intent    : {understanding.get('intent')}
  topic     : {understanding.get('topic')}
  urgency   : {understanding.get('urgency')}
  severity  : {understanding.get('severity')}
  edge_cases: {understanding.get('edge_cases') or 'none'}

Is there an open, unresolved action the support team must take for this customer right now?"""

    try:
        response = _llm_ticket.invoke([
            SystemMessage(content=_TICKET_ASSESSMENT_SYSTEM),
            HumanMessage(content=context),
        ])

        raw = re.sub(r"```json|```", "", response.content.strip()).strip()
        result = json.loads(raw)

        ticket_needed = bool(result.get("ticket_needed", False))
        reason = result.get("reason", "")
        print(f"[Decision] Call2 ticket assessment → needed={ticket_needed} | {reason}")
        return ticket_needed

    except Exception as exc:
        print(f"[Decision] Call2 ticket assessment failed ({exc}) — defaulting to no ticket.")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Hard-rule fallback — escalation decision ONLY
# Used only when Call 1 fails entirely.
# Operates purely on structured pipeline signals — zero text matching.
# ─────────────────────────────────────────────────────────────────────────────

def _hard_rule_escalation_only(understanding: dict) -> bool:
    """
    Returns True if escalation is needed, False if AI can respond.

    Uses ONLY structured signals produced by understanding_agent and
    edge_case_agent upstream — zero raw text analysis here.
    These signals are language-agnostic because the upstream agents already
    handled the language understanding.

    The ticket sub-decision is deliberately NOT made here.
    Call 2 (_assess_ticket_need) handles that. If Call 2 also fails,
    the default is "escalate" (no ticket) — always the safer production choice.
    """
    edge_cases = understanding.get("edge_cases", [])
    signals    = understanding.get("signals", {})
    intent     = understanding.get("intent", "")
    urgency    = understanding.get("urgency", "low")
    severity   = understanding.get("severity", "low")
    confidence = float(understanding.get("confidence", 0.5))

    return bool(
        understanding.get("needs_human_review")
        or severity == "high"
        or urgency == "high"
        or confidence < ESCALATION_THRESHOLD
        or signals.get("contradiction_detected")
        or "security_sensitive_feedback" in edge_cases
        or "billing_duplicate_charge" in edge_cases
        or "empty_input" in edge_cases
        or "too_long_input" in edge_cases
        or (intent in ("bug_report", "complaint") and urgency == "medium" and severity != "low")
        or ("too_short_input" in edge_cases and confidence <= 0.4)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def decide(understanding: dict) -> str:
    """
    Returns one of: 'respond' | 'escalate' | 'escalate_with_ticket'

    Execution flow:

      Normal path (Call 1 succeeds with a tool call):
        route_to_respond        → return 'respond'
        route_to_escalate       → return 'escalate'
        route_to_escalate_ticket→ run Call 2 to confirm → return final route

      Degraded path (Call 1 succeeds but returns no tool call):
        Hard rules → escalate/respond split
        Call 2     → ticket sub-decision
        Combined   → return final route

      Failure path (Call 1 throws an exception):
        Hard rules → escalate/respond split
        Call 2     → ticket sub-decision (best-effort)
        Combined   → return final route
        If Call 2 also fails: return 'escalate' (no ticket) — safe default
    """
    edge_cases        = understanding.get("edge_cases", [])
    signals           = understanding.get("signals", {})
    original_feedback = (
        understanding.get("original_feedback")
        or understanding.get("human_readable", "")
    )
    human_readable    = understanding.get("human_readable", "")

    call1_context = f"""Customer's actual feedback:
"{original_feedback}"

Plain-English situation summary (what they actually mean):
"{human_readable}"

Pipeline analysis signals:
  intent                : {understanding.get('intent')}
  sentiment             : {understanding.get('sentiment')}
  topic                 : {understanding.get('topic')}
  urgency               : {understanding.get('urgency')}
  severity              : {understanding.get('severity')}
  confidence            : {understanding.get('confidence')}
  needs_human_review    : {understanding.get('needs_human_review')}
  review_reason         : {understanding.get('review_reason') or 'none'}
  edge_cases            : {edge_cases if edge_cases else 'none'}
  contradiction_detected: {signals.get('contradiction_detected', False)}

Read the customer's actual situation carefully before choosing a tool."""

    # ── Call 1: Full routing decision ─────────────────────────────────────────
    try:
        response: AIMessage = _llm_router_with_tools.invoke([
            SystemMessage(content=_ROUTING_SYSTEM),
            HumanMessage(content=call1_context),
        ])

        if response.tool_calls:
            tc   = response.tool_calls[0]
            name = tc["name"]
            args = tc["args"]

            if name == "route_to_respond":
                print(f"[Decision] Call1 → respond | {args.get('reason', '')}")
                return "respond"

            if name == "route_to_escalate":
                print(f"[Decision] Call1 → escalate | {args.get('reason', '')} "
                      f"priority={args.get('priority', 'medium')}")
                return "escalate"

            if name == "route_to_escalate_ticket":
                # Call 1 says ticket needed — run Call 2 to independently confirm
                print(f"[Decision] Call1 → escalate+ticket candidate | "
                      f"{args.get('reason', '')} priority={args.get('priority', 'medium')}")
                ticket_confirmed = _assess_ticket_need(
                    original_feedback, human_readable, understanding
                )
                route = "escalate_with_ticket" if ticket_confirmed else "escalate"
                print(f"[Decision] Call2 confirmed → {route}")
                return route

        # Call 1 returned no tool call — fall through to hard rules + Call 2
        print("[Decision] Call1 returned no tool call — falling through to hard rules + Call2.")

    except Exception as exc:
        print(f"[Decision] Call1 failed ({exc}) — falling through to hard rules + Call2.")

    # ── Hard rules: escalate or respond? ─────────────────────────────────────
    needs_escalation = _hard_rule_escalation_only(understanding)

    if not needs_escalation:
        print("[Decision] Hard rules → respond")
        return "respond"

    # Escalation confirmed by hard rules — now ask Call 2 about the ticket
    print("[Decision] Hard rules → escalation needed, running Call2 for ticket decision.")
    ticket_needed = _assess_ticket_need(original_feedback, human_readable, understanding)
    route = "escalate_with_ticket" if ticket_needed else "escalate"
    print(f"[Decision] Final route (hard rules + Call2) → {route}")
    return route