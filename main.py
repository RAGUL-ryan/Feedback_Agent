from agents.ingestion_agent import ingest
from agents.understanding_agent import understand
from agents.context_agent import get_context
from agents.decision_agent import decide
from agents.edge_case_agent import apply_edge_case_rules
from agents.response_agent import generate_response
from agents.escalation_agent import escalate
from agents.learning_agent import learn

def run_feedback_pipeline(raw_feedback: str):
    print("\n== FEEDBACK AGENT PIPELINE ==")

    # 1. Ingest + clean
    ingested = ingest(raw_feedback)
    cleaned = ingested.get("cleaned_text", raw_feedback)
    detected_type = ingested.get("detected_type", "text")
    emoji_sentiment = ingested.get("emoji_sentiment", "neutral")
    print(f"[Ingestion] Type={detected_type} | {cleaned}")

    # 2. Understand
    analysis = understand(cleaned)
    analysis = apply_edge_case_rules(raw_feedback, cleaned, analysis)
    human_readable = analysis.get("human_readable", cleaned)
    emoji_detected = analysis.get("emoji_detected", False)
    print(f"[Understanding] {analysis}")

    # 3. RAG Context
    context = get_context(cleaned)
    print(f"[Context] Retrieved {len(context)} chars from knowledge base")

    # 4. Decide
    route = decide(analysis)
    print(f"[Decision] Route → {route}")

    if route == "respond":
        reply = generate_response(
            feedback=cleaned,
            analysis=analysis,
            context=context,
            original_input=raw_feedback,
            human_readable=human_readable
        )
        print(f"[Response] {reply}")
        learn(cleaned, reply, outcome="auto_resolved")
        return {
            "status": "replied",
            "response": reply,
            "understanding": analysis,
            "cleaned_text": cleaned,
            "original_input": raw_feedback,
            "detected_type": detected_type,
            "emoji_detected": emoji_detected,
            "human_readable": human_readable,
            "emoji_sentiment": emoji_sentiment
        }
    else:
        case = escalate(cleaned, analysis)
        case["topic"] = analysis.get("topic", "general")
        case["urgency"] = analysis.get("urgency", "medium")
        case["confidence"] = analysis.get("confidence", 0.5)
        return {
            "status": "escalated",
            "case": case,
            "understanding": analysis,
            "cleaned_text": cleaned,
            "original_input": raw_feedback,
            "detected_type": detected_type,
            "emoji_detected": emoji_detected,
            "human_readable": human_readable,
            "emoji_sentiment": emoji_sentiment
        }

if __name__ == "__main__":
    tests = [
        "😡😡 charged twice!!",
        "❤️🌟🌟🌟🌟🌟 love this app",
        "🤔 how do I reset password?",
        "🐛💥 app crashes on login",
        "⭐⭐ very disappointed with delivery"
    ]
    for t in tests:
        result = run_feedback_pipeline(t)
        print(f"\nInput: {t}")
        print(f"Readable: {result['human_readable']}")
        print(f"Status: {result['status']}\n")
