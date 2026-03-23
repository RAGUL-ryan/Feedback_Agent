from config import ESCALATION_THRESHOLD

def decide(understanding: dict) -> str:
    """
    Returns 'respond' or 'escalate' based on analysis.
    """
    confidence = understanding.get("confidence", 0.5)
    urgency = understanding.get("urgency", "low")
    intent = understanding.get("intent", "")

    if urgency == "high":
        return "escalate"
    if confidence < ESCALATION_THRESHOLD:
        return "escalate"
    if intent in ["bug_report", "complaint"] and urgency == "medium":
        return "escalate"
    return "respond"
