from typing import Any, Dict, List, Optional


HIGH_RISK_TERMS = {
    "fraud",
    "scam",
    "hacked",
    "hack",
    "security",
    "compromised",
    "data leak",
    "charged twice",
    "double charged",
}

QUESTION_TERMS = {"how", "what", "why", "where", "when", "can", "could", "help"}
AMBIGUOUS_TERMS = {"ok", "fine", "hmm", "maybe", "idk"}


def _normalize(text: str) -> str:
    normalized = (text or "").strip().lower()
    replacements = {
        "asap": "as soon as possible",
        "pls": "please",
        "plz": "please",
        "thx": "thanks",
        "idk": "i do not know",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return " ".join(normalized.split())


def _extract_star_rating(text: str) -> Optional[int]:
    if not text:
        return None

    star_count = text.count("⭐") + text.count("★")
    if 1 <= star_count <= 5:
        return star_count

    lowered = text.lower()
    for marker in ("1/5", "2/5", "3/5", "4/5", "5/5", "1 star", "2 star", "3 star", "4 star", "5 star", "1 stars", "2 stars", "3 stars", "4 stars", "5 stars"):
        if marker in lowered:
            return int(marker[0])
    return None


def _rating_sentiment(rating: Optional[int]) -> Optional[str]:
    if rating is None:
        return None
    if rating >= 4:
        return "positive"
    if rating <= 2:
        return "negative"
    return "neutral"


def apply_edge_case_rules(raw_feedback: str, cleaned_text: str, understanding: Dict[str, Any]) -> Dict[str, Any]:
    combined_text = f"{raw_feedback or ''} {cleaned_text or ''}".strip()
    normalized = _normalize(combined_text)
    rating = _extract_star_rating(raw_feedback or "") or _extract_star_rating(cleaned_text or "")
    rating_sentiment = _rating_sentiment(rating)
    model_sentiment = understanding.get("sentiment", "neutral")

    edge_cases: List[str] = []
    contradiction = False

    if rating_sentiment and model_sentiment != "neutral" and rating_sentiment != model_sentiment:
        contradiction = True
        edge_cases.append("rating_text_sentiment_mismatch")

    if "?" in combined_text and understanding.get("intent") in {"complaint", "bug_report"}:
        edge_cases.append("question_with_complaint")

    unique_words = {word.strip(".,!?") for word in normalized.split() if word.strip(".,!?")}
    if not unique_words or unique_words.issubset(AMBIGUOUS_TERMS):
        edge_cases.append("very_short_or_ambiguous_feedback")

    if "charged twice" in normalized or "double charged" in normalized:
        edge_cases.append("billing_duplicate_charge")
    if any(term in normalized for term in {"hack", "hacked", "security", "compromised", "fraud", "scam", "data leak"}):
        edge_cases.append("security_sensitive_feedback")

    final_severity = understanding.get("severity", "low")
    final_urgency = understanding.get("urgency", "low")
    final_confidence = float(understanding.get("confidence", 0.5))
    needs_human_review = bool(understanding.get("needs_human_review", False))
    review_reason = str(understanding.get("review_reason", "") or "")

    if any(term in normalized for term in HIGH_RISK_TERMS):
        final_severity = "high"
        final_urgency = "high"
        needs_human_review = True
        review_reason = "high_risk_feedback"

    if contradiction:
        final_confidence = min(final_confidence, 0.45)
        needs_human_review = True
        review_reason = "rating_text_sentiment_mismatch"

    if "very_short_or_ambiguous_feedback" in edge_cases:
        final_confidence = min(final_confidence, 0.4)
        needs_human_review = True
        if not review_reason:
            review_reason = "low_context_feedback"

    if final_confidence < 0.55 and not needs_human_review:
        needs_human_review = True
        review_reason = review_reason or "low_confidence_classification"

    merged = dict(understanding)
    merged.update(
        {
            "severity": final_severity,
            "urgency": final_urgency,
            "needs_human_review": needs_human_review,
            "review_reason": review_reason,
            "confidence": final_confidence,
            "edge_cases": edge_cases,
            "signals": {
                "star_rating": rating,
                "rating_sentiment": rating_sentiment,
                "model_sentiment": model_sentiment,
                "contradiction_detected": contradiction,
                "needs_human_review": needs_human_review,
            },
        }
    )

    if contradiction:
        merged["human_readable"] = (
            f"{understanding.get('human_readable', cleaned_text)} "
            "The rating and written comment conflict, so this should be reviewed carefully."
        ).strip()

    return merged
