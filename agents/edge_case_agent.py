import re
from typing import Any, Dict, List, Optional


NEGATIVE_TERMS = {
    "bad",
    "worst",
    "terrible",
    "awful",
    "poor",
    "hate",
    "broken",
    "bug",
    "bugs",
    "issue",
    "issues",
    "problem",
    "problems",
    "crash",
    "crashes",
    "crashed",
    "failing",
    "fail",
    "slow",
    "delay",
    "late",
    "useless",
    "disappointed",
    "frustrating",
    "angry",
    "refund",
    "charged",
    "charge",
    "double",
    "twice",
    "fraud",
    "scam",
    "hack",
    "hacked",
    "down",
    "error",
}

POSITIVE_TERMS = {
    "good",
    "great",
    "excellent",
    "awesome",
    "love",
    "loved",
    "perfect",
    "nice",
    "smooth",
    "helpful",
    "happy",
    "amazing",
    "best",
}

QUESTION_TERMS = {
    "how",
    "what",
    "why",
    "where",
    "when",
    "can",
    "could",
    "help",
}

SUGGESTION_TERMS = {
    "should",
    "wish",
    "could add",
    "please add",
    "suggest",
    "feature request",
    "would be better",
}

BUG_TERMS = {
    "bug",
    "bugs",
    "error",
    "errors",
    "crash",
    "crashes",
    "crashed",
    "not working",
    "broken",
    "stuck",
    "freeze",
    "frozen",
}

HIGH_URGENCY_TERMS = {
    "charged twice",
    "double charged",
    "fraud",
    "scam",
    "hacked",
    "hack",
    "security",
    "data leak",
    "compromised",
    "urgent",
    "immediately",
    "asap",
}

TOPIC_KEYWORDS = {
    "billing": {"charged", "charge", "refund", "payment", "invoice", "billing", "subscription"},
    "support": {"support", "agent", "response", "reply", "wait"},
    "delivery": {"delivery", "shipping", "order", "package", "dispatch", "tracking"},
    "product": {"app", "feature", "dashboard", "login", "password", "account", "api", "integration"},
}

POSITIVE_PATTERNS = (
    r"\bnot bad\b",
    r"\bno issue(?:s)?\b",
    r"\bworks? well\b",
    r"\bvery satisfied\b",
    r"\breally satisfied\b",
    r"\bquite happy\b",
    r"\blove (?:this|the)? ?(?:app|product|service|platform)?\b",
)

NEGATIVE_PATTERNS = (
    r"\bdid not like\b",
    r"\bdidn't like\b",
    r"\bdo not like\b",
    r"\bdon't like\b",
    r"\bnot like\b",
    r"\bnot liking\b",
    r"\bnot happy\b",
    r"\bnot satisfied\b",
    r"\bnot good\b",
    r"\bno good\b",
    r"\bnot useful\b",
    r"\bnot helpful\b",
    r"\bwrong\b",
    r"\bvery bad\b",
    r"\breally bad\b",
    r"\bpoor service\b",
    r"\bbad service\b",
    r"\bbad product\b",
    r"\bdoes not work\b",
    r"\bdoesn't work\b",
    r"\bnot working\b",
    r"\bcan't login\b",
    r"\bcannot login\b",
    r"\bfailed to\b",
    r"\bmoney (?:taken|deducted)\b",
    r"\bcharged twice\b",
    r"\bdouble charged\b",
    r"\bno result\b",
    r"\bno results\b",
    r"\bnot getting (?:any )?result(?:s)?\b",
    r"\bnot getting (?:any )?output\b",
    r"\bno output\b",
    r"\bno response\b",
    r"\bnothing happens\b",
    r"\bbut no (?:result|results|output|response)\b",
    r"\bworking but no (?:result|results|output|response)\b",
    r"\bworks? but no (?:result|results|output|response)\b",
)

BUG_PATTERNS = (
    r"\bapp crashes?\b",
    r"\bkeeps crashing\b",
    r"\bnot working\b",
    r"\bdoes not work\b",
    r"\bdoesn't work\b",
    r"\bwon't open\b",
    r"\bstuck\b",
    r"\bfrozen?\b",
    r"\berror\b",
    r"\bno result\b",
    r"\bno output\b",
    r"\bnothing happens\b",
)

QUESTION_PATTERNS = (
    r"\bhow do i\b",
    r"\bhow can i\b",
    r"\bwhere can i\b",
    r"\bwhat is\b",
    r"\bwhy (?:is|was|did|can't|cannot)\b",
    r"\bcan you help\b",
)

SUGGESTION_PATTERNS = (
    r"\byou should\b",
    r"\bi wish\b",
    r"\bplease add\b",
    r"\bit would be better\b",
    r"\bfeature request\b",
)

HIGH_URGENCY_PATTERNS = (
    r"\bcharged twice\b",
    r"\bdouble charged\b",
    r"\bhacked\b",
    r"\bcompromised\b",
    r"\bsecurity\b",
    r"\bdata leak\b",
    r"\burgent\b",
    r"\basap\b",
    r"\bimmediately\b",
    r"\bno access after payment\b",
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _extract_star_rating(text: str) -> Optional[int]:
    if not text:
        return None

    star_count = text.count("⭐") + text.count("★")
    if 1 <= star_count <= 5:
        return star_count

    patterns = [
        r"\b([1-5])\s*/\s*5\b",
        r"\b([1-5])\s*stars?\b",
        r"\b([1-5])\s*star\b",
        r"\b([1-5])\s*rating\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _contains_phrase(text: str, phrases: set[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", text)


def _match_patterns(text: str, patterns: tuple[str, ...]) -> List[str]:
    return [pattern for pattern in patterns if re.search(pattern, text, flags=re.IGNORECASE)]


def _score_sentiment(text: str) -> Dict[str, int]:
    normalized = _normalize(text)
    positive = 0
    negative = 0

    positive_matches = _match_patterns(normalized, POSITIVE_PATTERNS)
    negative_matches = _match_patterns(normalized, NEGATIVE_PATTERNS)

    positive += len(positive_matches) * 2
    negative += len(negative_matches) * 2

    for match in positive_matches:
        normalized = re.sub(match, " ", normalized, flags=re.IGNORECASE)
    for match in negative_matches:
        normalized = re.sub(match, " ", normalized, flags=re.IGNORECASE)

    # Common complaint structure: "works/working ... but/no ..."
    if re.search(r"\b(?:work|working|works)\b.*\b(?:but|no|not)\b", normalized, flags=re.IGNORECASE):
        negative += 2

    tokens = _tokenize(normalized)
    for token in tokens:
        if token in POSITIVE_TERMS:
            positive += 1
        if token in NEGATIVE_TERMS:
            negative += 1

    return {
        "positive": positive,
        "negative": negative,
        "positive_pattern_hits": len(positive_matches),
        "negative_pattern_hits": len(negative_matches),
    }


def _rating_sentiment(rating: Optional[int]) -> Optional[str]:
    if rating is None:
        return None
    if rating >= 4:
        return "positive"
    if rating <= 2:
        return "negative"
    return "neutral"


def _detect_topic(text: str, fallback: str) -> str:
    normalized = _normalize(text)
    if _match_patterns(normalized, BUG_PATTERNS):
        return "product"
    for topic, keywords in TOPIC_KEYWORDS.items():
        if _contains_phrase(normalized, keywords):
            return topic
    return fallback or "general"


def _resolve_intent(text: str, fallback: str, negative_score: int) -> str:
    normalized = _normalize(text)

    if _contains_phrase(normalized, BUG_TERMS) or _match_patterns(normalized, BUG_PATTERNS):
        return "bug_report"
    if "?" in text or any(token in normalized.split() for token in QUESTION_TERMS) or _match_patterns(normalized, QUESTION_PATTERNS):
        return "question"
    if _contains_phrase(normalized, SUGGESTION_TERMS) or _match_patterns(normalized, SUGGESTION_PATTERNS):
        return "suggestion"
    if negative_score > 0:
        return "complaint"
    if fallback:
        return fallback
    return "praise"


def _resolve_urgency(text: str, existing: str, intent: str, negative_score: int) -> str:
    normalized = _normalize(text)
    if _contains_phrase(normalized, HIGH_URGENCY_TERMS) or _match_patterns(normalized, HIGH_URGENCY_PATTERNS):
        return "high"
    if intent == "bug_report" and negative_score > 0:
        return "high"
    if intent in {"complaint", "bug_report"} and negative_score > 0:
        return "medium"
    return existing or "low"


def apply_edge_case_rules(raw_feedback: str, cleaned_text: str, understanding: Dict[str, Any]) -> Dict[str, Any]:
    combined_text = f"{raw_feedback or ''} {cleaned_text or ''}".strip()
    normalized = _normalize(combined_text)
    rating = _extract_star_rating(raw_feedback or "") or _extract_star_rating(cleaned_text or "")
    text_scores = _score_sentiment(combined_text)
    rating_sentiment = _rating_sentiment(rating)
    text_sentiment = "neutral"
    if text_scores["negative"] > text_scores["positive"]:
        text_sentiment = "negative"
    elif text_scores["positive"] > text_scores["negative"]:
        text_sentiment = "positive"

    edge_cases: List[str] = []
    contradiction = False

    if rating_sentiment and text_sentiment != "neutral" and rating_sentiment != text_sentiment:
        contradiction = True
        edge_cases.append("rating_text_sentiment_mismatch")

    if rating == 5 and text_scores["negative"] > 0:
        edge_cases.append("high_rating_with_negative_comment")
    if rating == 1 and text_scores["positive"] > 0:
        edge_cases.append("low_rating_with_positive_comment")
    if "?" in combined_text and text_scores["negative"] > 0:
        edge_cases.append("question_with_complaint")
    if text_scores["positive"] > 0 and text_scores["negative"] > 0:
        edge_cases.append("mixed_sentiment_feedback")
    unique_tokens = set(_tokenize(normalized))
    if not unique_tokens or unique_tokens.issubset({"ok", "fine", "hmm", "bad", "good"}):
        edge_cases.append("very_short_or_ambiguous_feedback")
    if any(term in normalized for term in {"charged twice", "double charged"}) or _match_patterns(normalized, (r"\bmoney (?:taken|deducted)\b",)):
        edge_cases.append("billing_duplicate_charge")
    if any(term in normalized for term in {"hack", "hacked", "security", "compromised"}) or _match_patterns(normalized, (r"\baccount (?:hacked|compromised)\b",)):
        edge_cases.append("security_sensitive_feedback")

    final_sentiment = understanding.get("sentiment", "neutral")
    if contradiction:
        final_sentiment = "negative" if text_scores["negative"] >= text_scores["positive"] else "neutral"
    elif text_sentiment != "neutral":
        final_sentiment = text_sentiment

    final_intent = _resolve_intent(combined_text, understanding.get("intent", ""), text_scores["negative"])
    final_topic = _detect_topic(combined_text, understanding.get("topic", "general"))
    final_urgency = _resolve_urgency(combined_text, understanding.get("urgency", "low"), final_intent, text_scores["negative"])
    final_confidence = float(understanding.get("confidence", 0.5))
    if contradiction:
        final_confidence = min(final_confidence, 0.45)
    if "very_short_or_ambiguous_feedback" in edge_cases:
        final_confidence = min(final_confidence, 0.4)

    merged = dict(understanding)
    merged.update({
        "sentiment": final_sentiment,
        "intent": final_intent,
        "topic": final_topic,
        "urgency": final_urgency,
        "confidence": final_confidence,
        "edge_cases": edge_cases,
        "signals": {
            "star_rating": rating,
            "rating_sentiment": rating_sentiment,
            "text_sentiment": text_sentiment,
            "positive_terms": text_scores["positive"],
            "negative_terms": text_scores["negative"],
            "positive_pattern_hits": text_scores["positive_pattern_hits"],
            "negative_pattern_hits": text_scores["negative_pattern_hits"],
            "contradiction_detected": contradiction,
        },
    })

    if contradiction:
        merged["human_readable"] = (
            f"{understanding.get('human_readable', cleaned_text)} "
            f"The rating and written comment conflict, so the written complaint should be prioritized."
        ).strip()

    return merged
