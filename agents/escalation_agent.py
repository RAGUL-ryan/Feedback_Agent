def escalate(feedback: str, understanding: dict) -> dict:
    """
    Packages the case for human team handoff.
    In production: send to Slack, email, or a ticketing system (Zendesk, Jira).
    """
    case = {
        "status": "escalated",
        "feedback": feedback,
        "reason": understanding.get("review_reason") or f"Urgency={understanding.get('urgency')}, Severity={understanding.get('severity')}, Confidence={understanding.get('confidence')}",
        "intent": understanding.get("intent"),
        "sentiment": understanding.get("sentiment"),
        "severity": understanding.get("severity"),
        "urgency": understanding.get("urgency"),
        "assigned_to": "human_support_team"
    }
    # TODO: plug in tools/notification.py here for real alerts
    print(f"[ESCALATION] Case forwarded to human team:\n{case}")
    return case
