from agents.ingestion_agent import ingest
from agents.understanding_agent import understand
from agents.context_agent import get_context
from agents.decision_agent import decide
from agents.edge_case_agent import apply_edge_case_rules
from agents.response_agent import generate_response
from agents.escalation_agent import escalate
from agents.learning_agent import learn
from agents.translation_agent import (
    is_tanglish,
    normalize_tanglish,
    translate_text,
    get_language_name,
)


def run_feedback_pipeline(raw_feedback: str, target_language: str = "en"):
    print("\n== FEEDBACK AGENT PIPELINE ==")
    print(f"[Language] Target -> {get_language_name(target_language)} ({target_language})")

    # 0. Tanglish detection
    tanglish_detected = is_tanglish(raw_feedback)
    if tanglish_detected:
        normalised_feedback = normalize_tanglish(raw_feedback)
        print(f"[Tanglish] Detected. Normalised: {normalised_feedback}")
    else:
        normalised_feedback = raw_feedback

    # 1. Ingest + clean
    ingested = ingest(normalised_feedback)
    cleaned = ingested.get("cleaned_text", normalised_feedback)
    detected_type = ingested.get("detected_type", "text")
    emoji_sentiment = ingested.get("emoji_sentiment", "neutral")
    print(f"[Ingestion] Type={detected_type} | {cleaned}")

    # 2. Understand
    analysis = understand(cleaned)
    analysis = apply_edge_case_rules(normalised_feedback, cleaned, analysis)
    human_readable = analysis.get("human_readable", cleaned)
    emoji_detected = analysis.get("emoji_detected", False)
    print(f"[Understanding] {analysis}")

    # 3. RAG Context
    context = get_context(human_readable or cleaned)
    print(f"[Context] Retrieved {len(context)} chars from knowledge base")

    # 4. Decide
    route = decide(analysis)
    print(f"[Decision] Route -> {route}")

    # 5. Build base payload
    base_payload = {
        "understanding": analysis,
        "cleaned_text": cleaned,
        "original_input": raw_feedback,
        "detected_type": detected_type,
        "emoji_detected": emoji_detected,
        "human_readable": human_readable,
        "emoji_sentiment": emoji_sentiment,
        "tanglish_detected": tanglish_detected,
        "target_language": target_language,
        "language_name": get_language_name(target_language),
    }

    if route == "respond":
        # Generate English reply first
        reply_en = generate_response(
            feedback=cleaned,
            analysis=analysis,
            context=context,
            original_input=normalised_feedback,
            human_readable=human_readable,
        )
        print(f"[Response EN] {reply_en}")

        # Translate to target language
        reply_translated = translate_text(reply_en, target_language)
        if target_language != "en":
            print(f"[Response {target_language}] {reply_translated}")

        learn(cleaned, reply_en, outcome="auto_resolved")

        return {
            "status": "replied",
            "response": reply_translated,
            "response_en": reply_en,
            **base_payload,
        }

    else:
        case = escalate(cleaned, analysis)
        case["topic"] = analysis.get("topic", "general")
        case["urgency"] = analysis.get("urgency", "medium")
        case["confidence"] = analysis.get("confidence", 0.5)

        escalation_message_en = (
            "Your feedback has been escalated to our human support team. "
            "They will review your case and contact you shortly."
        )
        escalation_message = translate_text(escalation_message_en, target_language)

        return {
            "status": "escalated",
            "case": case,
            "escalation_message": escalation_message,
            "escalation_message_en": escalation_message_en,
            **base_payload,
        }


if __name__ == "__main__":
    tests = [
        ("app very worst", "hi"),
        ("charged twice this month", "ta"),
        ("app romba mosam, work agala", "ta"),
        ("how do I reset my password?", "te"),
    ]
    for feedback, lang in tests:
        result = run_feedback_pipeline(feedback, target_language=lang)
        print(f"\nInput   : {feedback}")
        print(f"Language: {result['language_name']}")
        print(f"Tanglish: {result['tanglish_detected']}")
        print(f"Status  : {result['status']}")
        if result["status"] == "replied":
            print(f"Reply   : {result['response']}")