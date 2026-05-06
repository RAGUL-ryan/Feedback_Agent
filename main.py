"""
main.py  (replace your existing main.py)
─────────────────────────────────────────────────────────────────────────────
v5 changes (Approach 2 — authenticated users):
  • run_feedback_pipeline() accepts customer_name, customer_email, customer_phone
  • These come from the JWT token (decoded in app.py) — not from the form
  • Passed down to escalate() → crm_agent and email_agent
  • Auto-replied cases also log to CRM + send reply email (background thread)
"""

import threading
from concurrent.futures import ThreadPoolExecutor

from agents.ingestion_agent import ingest
from agents.understanding_agent import understand
from agents.context_agent import get_context
from agents.decision_agent import decide
from agents.edge_case_agent import apply_edge_case_rules
from agents.response_agent import generate_response
from agents.escalation_agent import escalate
from agents.learning_agent import learn
from agents.translation_agent import (
    detect_and_normalize,
    translate_text,
    get_language_name,
    supported_language_codes,
)


def _learn_async(cleaned: str, reply_en: str, outcome: str = "auto_resolved"):
    try:
        learn(cleaned, reply_en, outcome=outcome)
    except Exception as exc:
        print(f"[LEARNING] Background store failed: {exc}")


def _crm_and_email_reply(
    feedback: str,
    analysis: dict,
    reply_en: str,
    customer_name: str,
    customer_email: str,
    customer_phone: str,
):
    """Fire-and-forget: log auto-replied case to CRM + send reply email."""
    try:
        from agents.crm_agent import create_ticket
        create_ticket(
            feedback       = feedback,
            analysis       = analysis,
            status         = "replied",
            reply          = reply_en,
            customer_name  = customer_name,
            customer_email = customer_email,
            customer_phone = customer_phone,
        )
    except Exception as exc:
        print(f"[CRM] Auto-reply log failed (non-fatal): {exc}")

    try:
        from agents.email_agent import send_reply_email
        if customer_email:
            send_reply_email(
                customer_email = customer_email,
                customer_name  = customer_name,
                ai_reply       = reply_en,
                feedback       = feedback,
            )
    except Exception as exc:
        print(f"[EMAIL] Reply email failed (non-fatal): {exc}")


def run_feedback_pipeline(
    raw_feedback: str,
    target_language: str = "en",
    sector: str | None = None,
    customer_name: str = "Unknown",
    customer_email: str = "",
    customer_phone: str = "",
):
    from config import SECTOR as CONFIG_SECTOR
    effective_sector = (sector or CONFIG_SECTOR or "fintech").lower().strip()

    print("\n== FEEDBACK AGENT PIPELINE ==")
    print(f"[Sector]   {effective_sector.upper()}")
    print(f"[Customer] {customer_name} <{customer_email}>")
    print(f"[Lang]     Target → {get_language_name(target_language)} ({target_language})")

    # ── Stage A: Language detection ───────────────────────────────────────────
    lang_result         = detect_and_normalize(raw_feedback)
    tanglish_detected   = lang_result["tanglish_detected"]
    normalised_feedback = lang_result["normalized_feedback"]
    detected_language   = lang_result["detected_language"]

    # ── Stage B: ingest ║ understand (parallel) ───────────────────────────────
    with ThreadPoolExecutor(max_workers=2) as stage_b:
        future_ingest     = stage_b.submit(ingest, normalised_feedback)
        future_understand = stage_b.submit(understand, normalised_feedback, effective_sector)

    ingested        = future_ingest.result()
    analysis        = future_understand.result()
    cleaned         = ingested.get("cleaned_text", normalised_feedback)
    detected_type   = ingested.get("detected_type", "text")
    emoji_sentiment = ingested.get("emoji_sentiment", "neutral")

    # ── Instant reject: empty input ───────────────────────────────────────────
    if not normalised_feedback or not normalised_feedback.strip():
        reject_msg = "It looks like your message was empty. Please describe your issue and we will help you."
        return {
            "status": "rejected", "response": reject_msg,
            "response_en": reject_msg, "original_input": raw_feedback,
            "reason": "empty_input",
        }

    # ── Stage C: edge_case ║ get_context (parallel) ───────────────────────────
    human_readable_pre = analysis.get("human_readable", cleaned)
    with ThreadPoolExecutor(max_workers=2) as stage_c:
        future_edge    = stage_c.submit(apply_edge_case_rules, normalised_feedback, cleaned, analysis, effective_sector)
        future_context = stage_c.submit(get_context, human_readable_pre or cleaned)

    analysis = future_edge.result()
    context  = future_context.result()

    if analysis.get("instant_reject"):
        reject_msg = analysis.get("reject_message", "Please provide your feedback and we will help you.")
        return {
            "status": "rejected", "response": reject_msg,
            "response_en": reject_msg, "original_input": raw_feedback,
            "reason": "empty_input",
        }

    human_readable = analysis.get("human_readable", cleaned)
    emoji_detected = analysis.get("emoji_detected", False)

    # ── Stage D: Routing decision ─────────────────────────────────────────────
    route = decide(analysis)

    base_payload = {
        "understanding":     analysis,
        "cleaned_text":      cleaned,
        "original_input":    raw_feedback,
        "detected_type":     detected_type,
        "emoji_detected":    emoji_detected,
        "human_readable":    human_readable,
        "emoji_sentiment":   emoji_sentiment,
        "tanglish_detected": tanglish_detected,
        "detected_language": detected_language,
        "target_language":   target_language,
        "language_name":     get_language_name(target_language),
        "sector":            effective_sector,
        "customer_name":     customer_name,
    }

    # ── Stage E: Auto-respond ─────────────────────────────────────────────────
    if route == "respond":
        reply_en = generate_response(
            feedback       = cleaned,
            analysis       = analysis,
            context        = context,
            original_input = normalised_feedback,
            human_readable = human_readable,
            sector         = effective_sector,
        )
        print(f"[Response EN] {reply_en}")
        reply_translated = translate_text(reply_en, target_language)

        analysis_snapshot = {**analysis, "sector": effective_sector}

        # Background: learn + CRM log + reply email
        threading.Thread(target=_learn_async, args=(cleaned, reply_en), daemon=True).start()
        threading.Thread(
            target=_crm_and_email_reply,
            args=(cleaned, analysis_snapshot, reply_en, customer_name, customer_email, customer_phone),
            daemon=True,
        ).start()

        return {
            "status":      "replied",
            "response":    reply_translated,
            "response_en": reply_en,
            **base_payload,
        }

    # ── Stage E: Escalate ─────────────────────────────────────────────────────
    else:
        case = escalate(
            feedback       = cleaned,
            understanding  = analysis,
            sector         = effective_sector,
            customer_name  = customer_name,
            customer_email = customer_email,
            customer_phone = customer_phone,
        )

        escalation_message_en = (
            "Your feedback has been escalated to our human support team. "
            "They will review your case and contact you shortly. "
            f"A confirmation email has been sent to {customer_email or 'you'}."
        )
        escalation_message = translate_text(escalation_message_en, target_language)

        return {
            "status":                "escalated",
            "case":                  case,
            "escalation_message":    escalation_message,
            "escalation_message_en": escalation_message_en,
            "crm_ticket_id":         case.get("crm_ticket_id"),
            "crm_ticket_url":        case.get("crm_ticket_url"),
            **base_payload,
        }