"""
main.py
─────────────────────────────────────────────────────────────────────────────
v7 changes — Agent-driven CRM + email gating:

  decision_agent now returns one of THREE routes:
    'respond'              → AI auto-reply (unchanged)
    'escalate'             → human review, inline acknowledgment only
    'escalate_with_ticket' → human review + CRM ticket + customer email

  main.py passes create_ticket=True to escalation_agent only when the
  decision_agent returned 'escalate_with_ticket'. No other logic change.

  The decision of whether a ticket is warranted lives entirely in
  decision_agent.py. main.py is purely a router.
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
            "status":         "rejected",
            "response":       reject_msg,
            "response_en":    reject_msg,
            "original_input": raw_feedback,
            "reason":         "empty_input",
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
            "status":         "rejected",
            "response":       reject_msg,
            "response_en":    reject_msg,
            "original_input": raw_feedback,
            "reason":         "empty_input",
        }

    human_readable = analysis.get("human_readable", cleaned)
    emoji_detected = analysis.get("emoji_detected", False)

    # ── Stage D: Routing decision (respond / escalate / escalate_with_ticket) ─
    route = decide(analysis)
    print(f"[Pipeline] Route decision: {route}")

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
        "route":             route,
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

        # Background: store to RAG for future improvements
        threading.Thread(
            target=_learn_async,
            args=(cleaned, reply_en),
            daemon=True,
        ).start()

        return {
            "status":      "replied",
            "response":    reply_translated,
            "response_en": reply_en,
            **base_payload,
        }

    # ── Stage E: Escalate (with or without CRM ticket) ───────────────────────
    else:
        # decision_agent returns 'escalate_with_ticket' when a formal ticket
        # and customer email notification are warranted. Otherwise 'escalate'.
        create_ticket = (route == "escalate_with_ticket")

        case = escalate(
            feedback       = cleaned,
            understanding  = analysis,
            create_ticket  = create_ticket,
            sector         = effective_sector,
            customer_name  = customer_name,
            customer_email = customer_email,
            customer_phone = customer_phone,
        )

        ack_en         = case.get("acknowledgment", "We've received your feedback and will look into it shortly.")
        ack_translated = translate_text(ack_en, target_language)

        return {
            "status":         "escalated",
            "response":       ack_translated,
            "response_en":    ack_en,
            "ticket_created": create_ticket,
            "crm":            case.get("crm", {}),
            "email":          case.get("email", {}),
            "case":           case,
            **base_payload,
        }