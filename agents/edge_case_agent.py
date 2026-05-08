"""
agents/edge_case_agent.py
─────────────────────────────────────────────────────────────────────────────
True agentic edge-case detector.

v3 changes:
  • System prompt is now sector-aware — the LLM understands what "billing issue"
    or "security risk" means differently in fintech vs healthcare vs food, etc.
  • apply_edge_case_rules() accepts an optional `sector` param.
  • All existing tools and fallback logic preserved from v2.
"""

from __future__ import annotations
from typing import Any, Dict
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
#from langchain_groq import ChatGroq
from config import MODEL_NAME, SECTOR
from langchain_openai import ChatOpenAI

# ─────────────────────────────────────────────────────────────────────────────
# Tools  (narrow, deterministic — LLM decides when to call each)
# ─────────────────────────────────────────────────────────────────────────────

@tool
def check_empty_input(reason: str) -> dict:
    """
    Flag this submission as empty or effectively blank.
    Call when: input is empty, whitespace only, single punctuation, or no
    meaningful content whatsoever.
    reason: describe what was received.
    """
    return {
        "edge_case": "empty_input",
        "severity": "low",
        "urgency": "low",
        "confidence_cap": 0.0,
        "needs_human_review": False,
        "review_reason": f"empty_submission: {reason}",
        "instant_reject": True,
        "reject_message": (
            "It looks like your message was empty. "
            "Please describe your issue or question and we will be happy to help."
        ),
    }


@tool
def check_input_length(word_count: int, issue: str) -> dict:
    """
    Flag this input as problematic because of its length.
    Call when:
      word_count < 3  → issue = 'too_short'  (not enough context to help)
      word_count > 500 → issue = 'too_long'  (likely spam or multi-issue dump)
    word_count: approximate number of words.
    issue: 'too_short' or 'too_long'
    """
    if issue == "too_short":
        return {
            "edge_case": "too_short_input",
            "confidence_cap": 0.35,
            "needs_human_review": True,
            "review_reason": f"input_too_short ({word_count} words) — insufficient context",
        }
    return {
        "edge_case": "too_long_input",
        "confidence_cap": 0.5,
        "needs_human_review": True,
        "review_reason": f"input_too_long ({word_count} words) — possible spam or multi-issue",
    }


@tool
def extract_star_rating(text: str) -> dict:
    """
    Extract a numeric star rating (1–5) from text or emoji if present.
    Call whenever feedback might contain a star rating (⭐ emoji or "3/5" text).
    Returns the rating and its sentiment polarity.
    """
    count = text.count("⭐") + text.count("★")
    if 1 <= count <= 5:
        rating = count
    else:
        lowered = text.lower()
        rating = None
        for marker in (
            "1/5","2/5","3/5","4/5","5/5",
            "1 star","2 star","3 star","4 star","5 star",
            "1 stars","2 stars","3 stars","4 stars","5 stars",
        ):
            if marker in lowered:
                rating = int(marker[0])
                break
    if rating is None:
        return {"star_rating": None, "rating_sentiment": None}
    sentiment = "positive" if rating >= 4 else ("negative" if rating <= 2 else "neutral")
    return {"star_rating": rating, "rating_sentiment": sentiment}


@tool
def flag_security_risk(reason: str) -> dict:
    """
    Flag this feedback as security-sensitive needing immediate human review.
    Call for ANY hint of: unauthorized access, hacking, fraud, scam, data breach,
    account compromise, suspicious transactions — even indirect phrasing like
    "someone else is using my account" or "I didn't make this payment".
    Judge by INTENT, not exact keywords.
    reason: brief explanation of the security concern.
    """
    return {
        "edge_case": "security_sensitive_feedback",
        "severity": "high",
        "urgency": "high",
        "needs_human_review": True,
        "review_reason": f"security_risk: {reason}",
    }


@tool
def flag_billing_issue(reason: str) -> dict:
    """
    Flag this feedback as a billing or financial anomaly needing human review.
    Call for ANY financial concern: double charge, wrong amount, missing refund,
    unexpected deduction, overcharge — even indirect phrasing like
    "you took money twice" or "debited twice" or "extra amount deducted".
    reason: brief explanation of the billing concern.
    """
    return {
        "edge_case": "billing_duplicate_charge",
        "severity": "high",
        "urgency": "high",
        "needs_human_review": True,
        "review_reason": f"billing_anomaly: {reason}",
    }


@tool
def flag_ambiguous(reason: str) -> dict:
    """
    Flag this feedback as too vague to classify confidently.
    Call when: feedback has no clear intent, single ambiguous word/emoji with
    multiple interpretations, or completely unclear what the user wants.
    reason: explain why the feedback is ambiguous.
    """
    return {
        "edge_case": "very_short_or_ambiguous_feedback",
        "confidence_cap": 0.4,
        "needs_human_review": True,
        "review_reason": f"ambiguous_input: {reason}",
    }


@tool
def flag_contradiction(star_rating: int, text_sentiment: str, explanation: str) -> dict:
    """
    Flag a contradiction between a star rating and the text sentiment.
    Call when a high rating accompanies clearly negative text, or a low rating
    accompanies clearly positive text.
    star_rating: numeric rating (1–5).
    text_sentiment: what the text actually expresses — 'positive' or 'negative'.
    explanation: brief description of the contradiction.
    """
    return {
        "edge_case": "rating_text_sentiment_mismatch",
        "contradiction_detected": True,
        "confidence_cap": 0.45,
        "needs_human_review": True,
        "review_reason": f"contradiction: {explanation}",
        "human_readable_suffix": (
            "The star rating and written comment conflict — this needs careful review."
        ),
    }


@tool
def set_needs_human_review(reason: str) -> dict:
    """
    Mark this case for human review for any reason not covered by other tools.
    Use for: legal language, threats, emotional distress, regulatory mentions,
    multiple unrelated issues in one message, safety concerns.
    reason: clear explanation of why human judgment is needed.
    """
    return {
        "needs_human_review": True,
        "review_reason": reason,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sector-aware system prompt builder
# ─────────────────────────────────────────────────────────────────────────────

_SECTOR_CONTEXT = {
    "fintech": (
        "financial platform (banking, payments, UPI, wallet, loans). "
        "Security risks include: unauthorized transactions, account takeover, OTP fraud. "
        "Billing issues include: double debit, wrong EMI, missing refund, unexpected deduction."
    ),
    "education": (
        "online education platform (courses, classes, certifications). "
        "Security risks include: unauthorized account access, impersonation. "
        "Billing issues include: fee charged but access not given, unexpected deduction, refund not processed."
    ),
    "healthcare": (
        "healthcare / hospital platform (appointments, billing, patient portal). "
        "Security risks include: patient data breach, unauthorized medical record access. "
        "Billing issues include: overcharged for procedure, insurance claim rejected incorrectly. "
        "Also flag: any mention of medical emergency, wrong diagnosis, or patient safety."
    ),
    "food": (
        "food delivery / restaurant platform. "
        "Security risks include: account compromise, payment fraud. "
        "Billing issues include: double charged for order, wrong amount debited, refund not received. "
        "Also flag: food safety concerns (foreign object, allergic reaction) as high severity."
    ),
    "ecommerce": (
        "ecommerce / online shopping platform. "
        "Security risks include: account hacked, unauthorized purchase, data leak. "
        "Billing issues include: overcharged, double payment, discount not applied. "
        "Also flag: counterfeit products, dangerous/defective products as high severity."
    ),
}

_DEFAULT_SECTOR_CONTEXT = _SECTOR_CONTEXT["fintech"]


def _build_system_prompt(sector: str) -> str:
    context = _SECTOR_CONTEXT.get(sector, _DEFAULT_SECTOR_CONTEXT)
    return f"""You are a feedback edge-case detection agent for a {sector} customer support system.

This is a {context}

Read the customer feedback and the prior analysis, then call the appropriate tools
to flag any edge cases present. You may call MULTIPLE tools in one turn.

When to call each tool:
  check_empty_input      → submission is blank, whitespace, or has zero meaningful content
  check_input_length     → fewer than 3 words (too_short) OR more than 500 words (too_long)
  extract_star_rating    → text contains star symbols or rating text like "3/5" or "2 stars"
  flag_security_risk     → ANY hint of unauthorized access, fraud, account compromise,
                           data leak — even indirect. Judge by MEANING, not keywords.
  flag_billing_issue     → ANY financial anomaly — double charge, missing refund,
                           wrong deduction — even indirect phrasing.
  flag_ambiguous         → cannot determine what the user actually wants
  flag_contradiction     → star rating clearly disagrees with text emotion
  set_needs_human_review → legal threat, emotional distress, regulatory mention,
                           safety concern, or multiple unrelated issues in one message

Important: judge by MEANING and CONTEXT — not exact keywords.
If no edge cases exist, call NO tools."""


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

TOOLS = [
    check_empty_input,
    check_input_length,
    extract_star_rating,
    flag_security_risk,
    flag_billing_issue,
    flag_ambiguous,
    flag_contradiction,
    set_needs_human_review,
]

#_llm = ChatGroq(model=MODEL_NAME, temperature=0).bind_tools(TOOLS)


_llm = ChatOpenAI(model=MODEL_NAME, temperature=0)

def apply_edge_case_rules(
    raw_feedback: str,
    cleaned_text: str,
    understanding: Dict[str, Any],
    sector: str | None = None,
) -> Dict[str, Any]:
    """
    Run the LLM edge-case agent and merge results into the understanding dict.
    Falls back gracefully if the LLM call fails.
    """
    effective_sector = (sector or understanding.get("sector") or SECTOR or "fintech").lower()

    # ── Fast path: empty input — skip LLM call entirely ──────────────────────
    if not raw_feedback or not raw_feedback.strip():
        merged = dict(understanding)
        merged.update({
            "severity": "low", "urgency": "low",
            "needs_human_review": False,
            "review_reason": "empty_submission",
            "confidence": 0.0,
            "edge_cases": ["empty_input"],
            "instant_reject": True,
            "reject_message": (
                "It looks like your message was empty. "
                "Please describe your issue or question and we will be happy to help."
            ),
            "signals": {
                "star_rating": None, "rating_sentiment": None,
                "model_sentiment": understanding.get("sentiment", "neutral"),
                "contradiction_detected": False, "needs_human_review": False,
            },
        })
        return merged

    word_count = len(raw_feedback.split())

    user_message = f"""Raw feedback ({word_count} words): {raw_feedback}

Cleaned interpretation: {cleaned_text}

Prior analysis:
  sentiment : {understanding.get('sentiment')}
  intent    : {understanding.get('intent')}
  urgency   : {understanding.get('urgency')}
  confidence: {understanding.get('confidence')}
  topic     : {understanding.get('topic')}
  summary   : {understanding.get('human_readable')}

Identify and flag any edge cases using the available tools."""

    system_prompt = _build_system_prompt(effective_sector)
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]

    try:
        response: AIMessage = _llm.invoke(messages)
    except Exception as exc:
        print(f"[EdgeCase] LLM call failed ({exc}), skipping edge-case enrichment.")
        merged = dict(understanding)
        merged.setdefault("edge_cases", [])
        merged.setdefault("signals", {
            "star_rating": None, "rating_sentiment": None,
            "model_sentiment": understanding.get("sentiment", "neutral"),
            "contradiction_detected": False,
            "needs_human_review": bool(understanding.get("needs_human_review", False)),
        })
        return merged

    # ── Process tool calls ────────────────────────────────────────────────────
    edge_cases: list[str] = []
    signals: dict = {
        "star_rating": None, "rating_sentiment": None,
        "model_sentiment": understanding.get("sentiment", "neutral"),
        "contradiction_detected": False, "needs_human_review": False,
    }
    final_severity        = understanding.get("severity", "low")
    final_urgency         = understanding.get("urgency", "low")
    final_confidence      = float(understanding.get("confidence", 0.5))
    needs_human_review    = bool(understanding.get("needs_human_review", False))
    review_reason         = str(understanding.get("review_reason", "") or "")
    human_readable_suffix = ""
    instant_reject        = False
    reject_message        = ""

    tool_map = {t.name: t for t in TOOLS}

    for tc in (response.tool_calls or []):
        name, args = tc["name"], tc["args"]
        if name not in tool_map:
            continue
        try:
            result: dict = tool_map[name].invoke(args)
        except Exception as exc:
            print(f"[EdgeCase] Tool '{name}' failed ({exc}), skipping.")
            continue

        if "edge_case" in result and result["edge_case"] not in edge_cases:
            edge_cases.append(result["edge_case"])
        if result.get("severity") == "high":
            final_severity = "high"
        if result.get("urgency") == "high":
            final_urgency = "high"
        if result.get("needs_human_review"):
            needs_human_review = True
            if result.get("review_reason"):
                review_reason = result["review_reason"]
        if "confidence_cap" in result:
            final_confidence = min(final_confidence, result["confidence_cap"])
        if "star_rating" in result:
            signals["star_rating"] = result["star_rating"]
            signals["rating_sentiment"] = result["rating_sentiment"]
        if result.get("contradiction_detected"):
            signals["contradiction_detected"] = True
        if result.get("human_readable_suffix"):
            human_readable_suffix = result["human_readable_suffix"]
        if result.get("instant_reject"):
            instant_reject = True
            reject_message = result.get("reject_message", "")

    # Auto-flag borderline confidence
    if final_confidence < 0.55 and not needs_human_review:
        needs_human_review = True
        review_reason = review_reason or "low_confidence_classification"

    signals["needs_human_review"] = needs_human_review

    merged = dict(understanding)
    merged.update({
        "severity": final_severity,
        "urgency": final_urgency,
        "needs_human_review": needs_human_review,
        "review_reason": review_reason,
        "confidence": final_confidence,
        "edge_cases": edge_cases,
        "signals": signals,
    })
    if instant_reject:
        merged["instant_reject"] = True
        merged["reject_message"] = reject_message
    if human_readable_suffix:
        base = merged.get("human_readable", cleaned_text)
        merged["human_readable"] = f"{base} {human_readable_suffix}".strip()

    return merged