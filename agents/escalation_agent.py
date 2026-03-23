def escalate(feedback: str, understanding: dict) -> dict:
    """
    Packages the case for human team handoff.
    In production: send to Slack, email, or a ticketing system (Zendesk, Jira).
    """
    case = {
        "status": "escalated",
        "feedback": feedback,
        "reason": f"Urgency={understanding.get('urgency')}, Confidence={understanding.get('confidence')}",
        "intent": understanding.get("intent"),
        "sentiment": understanding.get("sentiment"),
        "assigned_to": "human_support_team"
    }
    # TODO: plug in tools/notification.py here for real alerts
    print(f"[ESCALATION] Case forwarded to human team:\n{case}")
    return case
