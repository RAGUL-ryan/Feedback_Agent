from config import ESCALATION_THRESHOLD

def decide(understanding: dict) -> str:
    """
    Returns 'respond' or 'escalate' based on analysis.
    """
    confidence = understanding.get("confidence", 0.5)
    urgency = understanding.get("urgency", "low")
    intent = understanding.get("intent", "")
    edge_cases = understanding.get("edge_cases", [])
    signals = understanding.get("signals", {})

    if urgency == "high":
        return "escalate"
    if confidence < ESCALATION_THRESHOLD:
        return "escalate"
    if signals.get("contradiction_detected"):
        return "escalate"
    if "security_sensitive_feedback" in edge_cases:
        return "escalate"
    if "billing_duplicate_charge" in edge_cases:
        return "escalate"
    if "very_short_or_ambiguous_feedback" in edge_cases and confidence <= 0.4:
        return "escalate"
    if intent in ["bug_report", "complaint"] and urgency == "medium":
        return "escalate"
    return "respond"
