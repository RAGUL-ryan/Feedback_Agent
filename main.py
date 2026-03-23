from agents.ingestion_agent import ingest
from agents.understanding_agent import understand
from agents.context_agent import get_context
from agents.decision_agent import decide
from agents.response_agent import generate_response
from agents.escalation_agent import escalate
from agents.learning_agent import learn

def run_feedback_pipeline(raw_feedback: str):
    print("\n== FEEDBACK AGENT PIPELINE ==")

    # 1. Ingest + clean
    ingested = ingest(raw_feedback)
    cleaned = ingested.get("cleaned_text", raw_feedback)
    print(f"[Ingestion] {cleaned}")

    # 2. Understand
    analysis = understand(cleaned)
    print(f"[Understanding] {analysis}")

    # 3. Fetch context via RAG
    context = get_context(cleaned)
    print(f"[Context] Retrieved {len(context)} chars from knowledge base")

    # 4. Decide routing
    route = decide(analysis)
    print(f"[Decision] Route → {route}")

    if route == "respond":
        # 5a. Generate response
        reply = generate_response(cleaned, analysis, context)
        print(f"[Response] {reply}")

        # 6. Learn from outcome
        learn(cleaned, reply, outcome="auto_resolved")
        return {
            "status": "replied",
            "response": reply,
            "understanding": analysis,
            "cleaned_text": cleaned
        }

    else:
        # 5b. Escalate
        case = escalate(cleaned, analysis)
        case["sentiment"] = analysis.get("sentiment")
        case["intent"] = analysis.get("intent")
        case["topic"] = analysis.get("topic")
        case["urgency"] = analysis.get("urgency")
        case["confidence"] = analysis.get("confidence")
        return {
            "status": "escalated",
            "case": case,
            "understanding": analysis,
            "cleaned_text": cleaned
        }


if __name__ == "__main__":
    # Test it
    sample = "I've been charged twice for my subscription this month and no one is helping me!"
    result = run_feedback_pipeline(sample)
    print("\n== RESULT ==")
    print(result)
