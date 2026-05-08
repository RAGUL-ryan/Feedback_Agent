"""
agents/understanding_agent.py
─────────────────────────────────────────────────────────────────────────────
Feedback understanding agent. Pure LLM call — analysis IS the reasoning task.

Key changes v3:
  • Sector-aware analysis: fintech, education, healthcare, food, ecommerce
  • Topic is now SEMANTIC, not keyword-matched.
    e.g. "app crashes" → topic = "app_performance", NOT "product"
    e.g. "delivery late" → topic = "delivery_logistics" in ecommerce
  • Each sector has its own intent + topic taxonomy so the LLM picks
    the most meaningful label for that domain — no hardcoded keyword matching.
  • severity / needs_human_review / review_reason carried forward from v2.
  • understand() now accepts an optional `sector` param; falls back to
    config.SECTOR, then to "fintech".
"""

import json
import re
from langchain_core.prompts import ChatPromptTemplate
#from langchain_groq import ChatGroq
from config import MODEL_NAME, SECTOR          # Add SECTOR = "fintech" to config.py

from langchain_openai import ChatOpenAI
_llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
#_llm = ChatGroq(model=MODEL_NAME, temperature=0)

# ─────────────────────────────────────────────────────────────────────────────
# Sector taxonomy definitions
# ─────────────────────────────────────────────────────────────────────────────

_SECTOR_TAXONOMY: dict[str, dict] = {

    "fintech": {
        "intents": """
  complaint       – unhappy with product, service, or experience
  praise          – expressing satisfaction or gratitude
  question        – needs information or help understanding something
  suggestion      – proposing a feature or improvement
  bug_report      – reporting a technical error or broken feature
  refund_request  – explicitly wants money back or a reversal
  fraud_report    – unauthorized transaction, suspicious activity, scam
  general         – cannot be clearly classified above""",

        "topics": """
  billing         – payments, charges, refunds, transaction history, wrong deductions
  account         – login, KYC, profile, limits, account access, blocked account
  app_performance – crashes, slow load, UI bugs, app not opening, feature broken in app
  security        – fraud, unauthorized access, OTP issues, password, data safety
  transfer        – money transfers, UPI, NEFT, wallet to bank, send money failure
  card            – debit/credit card issues, block/unblock, declined, card limits
  loan            – EMI, loan application, disbursement, repayment, interest query
  support         – customer care response time, agent behavior, ticket resolution
  general         – none of the above""",

        "severity_notes": """
  high   – fraud, unauthorized transaction, account locked, data breach, double charge, strong distress
  medium – app crash affecting transactions, billing confusion, login failure
  low    – UI suggestion, general question, minor delay, positive feedback""",
    },

    "education": {
        "intents": """
  complaint       – unhappy with course, teacher, or platform experience
  praise          – positive feedback about content, instructor, or platform
  question        – asking about syllabus, schedule, certification, or fees
  suggestion      – proposing new topics, features, or improvements
  bug_report      – technical issue with platform, video not loading, quiz broken
  enrollment      – wants to enroll, upgrade plan, or switch course
  refund_request  – wants fee refund or cancellation
  general         – cannot be clearly classified above""",

        "topics": """
  course_content  – quality, accuracy, depth, or relevance of course material
  instructor      – teacher clarity, pace, expertise, availability, or behavior
  app_performance – platform/app crashes, video buffering, quiz not loading, login broken
  certification   – certificates, completion badges, validity, issue delay
  fees_billing    – payment failure, refund, pricing, subscription renewal
  schedule        – class timing, batch, calendar, rescheduling, missed class
  enrollment      – joining a course, access denied after payment, waitlist
  support         – response time, help desk, agent behavior, unresolved ticket
  general         – none of the above""",

        "severity_notes": """
  high   – exam/assessment data lost, account locked, unauthorized fee deduction
  medium – videos not loading, course access denied after payment, instructor absent
  low    – schedule question, content suggestion, general praise, feature request""",
    },

    "healthcare": {
        "intents": """
  complaint       – unhappy with care quality, wait time, or staff behavior
  praise          – appreciating doctor, nurse, facility, or overall experience
  question        – asking about appointment, test results, prescriptions, or costs
  suggestion      – proposing service or facility improvements
  bug_report      – technical issue with the health app or patient portal
  appointment     – booking, rescheduling, or canceling an appointment
  refund_request  – medical bill dispute or insurance reimbursement issue
  emergency       – urgent medical concern or patient safety issue
  general         – cannot be clearly classified above""",

        "topics": """
  doctor_quality  – doctor diagnosis, behavior, communication, or expertise
  nurse_staff     – nursing care, attendant behavior, ward or reception staff
  app_performance – health app bugs, portal crashes, login issues, feature broken
  billing         – medical bills, insurance claims, payment issues, overcharging
  appointment     – scheduling, long waiting time, cancellation, no-show
  prescription    – medicine issues, dosage instructions, pharmacy availability
  test_reports    – lab results, scan reports, delay in report delivery
  hygiene         – cleanliness of facility, equipment sterilization, safety
  emergency_care  – urgent or critical care response time and quality
  support         – helpdesk, call center, complaint resolution time
  general         – none of the above""",

        "severity_notes": """
  high   – medical emergency, wrong diagnosis, serious safety risk, life-threatening concern
  medium – missed appointment, wrong billing, delayed reports, rude staff complaint
  low    – general question, minor suggestion, appreciation, minor scheduling query""",
    },

    "food": {
        "intents": """
  complaint       – unhappy with food quality, delivery, or service
  praise          – positive feedback about food, delivery, or restaurant
  question        – asking about menu, ingredients, timing, or active offers
  suggestion      – proposing new dishes, better packaging, or service improvements
  bug_report      – issue with the food ordering app or payment during order
  refund_request  – wants refund for wrong, missing, or unacceptable order
  order_issue     – wrong item delivered, item missing, order cancelled unexpectedly
  general         – cannot be clearly classified above""",

        "topics": """
  food_quality    – taste, freshness, portion size, temperature on arrival
  delivery        – delivery time, packaging, delivery partner behavior, cold food
  app_performance – ordering app crashes, payment failures, OTP issue, UI bugs
  hygiene         – foreign object in food, unclean packaging, safety concern
  pricing         – overcharge, incorrect bill, discount not applied, hidden fee
  restaurant      – specific restaurant behavior, ambiance, preparation quality
  offer_coupon    – promo code not applied, cashback not credited, offer mismatch
  support         – customer care for order issues, refund handling, complaint response
  general         – none of the above""",

        "severity_notes": """
  high   – food safety incident (foreign object, allergic reaction), unauthorized charge
  medium – wrong or missing order, app down during active order, late delivery
  low    – general feedback, minor taste preference, menu question, suggestion""",
    },

    "ecommerce": {
        "intents": """
  complaint       – unhappy with product, seller, or platform experience
  praise          – positive experience with product, seller, or delivery
  question        – asking about order status, return policy, or product details
  suggestion      – proposing catalog, feature, or UX improvements
  bug_report      – technical issue on the website or app
  refund_request  – wants refund or replacement for a received product
  return_request  – wants to initiate a product return
  order_issue     – order not delivered, wrong item sent, damaged on arrival
  general         – cannot be clearly classified above""",

        "topics": """
  product_quality – defective, damaged, counterfeit, or not-as-described product
  delivery        – shipping delay, wrong address delivered, lost package, courier issue
  app_performance – website/app crashes, checkout failure, search broken, login issue
  return_refund   – return/refund initiation, status, delay, or rejection
  pricing         – incorrect price shown, missing discount, overcharge at checkout
  seller          – third-party seller behavior, fake listing, incorrect product info
  payment         – payment failure, double charge, EMI conversion, wallet credit
  support         – customer care helpfulness, resolution time, escalation handling
  general         – none of the above""",

        "severity_notes": """
  high   – counterfeit product, unauthorized charge, safety-critical product defect
  medium – non-delivery, wrong item sent, refund not processed, app down at checkout
  low    – general question, minor UI suggestion, packaging feedback, praise""",
    },
}

_DEFAULT_SECTOR = "fintech"


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder — fully dynamic, no hardcoded labels in base prompt
# ─────────────────────────────────────────────────────────────────────────────

def _build_system_prompt(sector: str) -> str:
    taxonomy = _SECTOR_TAXONOMY.get(sector, _SECTOR_TAXONOMY[_DEFAULT_SECTOR])

    return f"""You are a senior customer feedback analyst for a {sector} platform.

Analyze the customer feedback and return a precise structured assessment.

━━━ SECTOR: {sector.upper()} ━━━

──── INTENT ────
Choose the ONE intent that best captures what the customer is trying to DO or EXPRESS:
{taxonomy["intents"]}

──── TOPIC ────
Choose the ONE topic that best describes the ROOT CAUSE of the feedback.
Topic must reflect MEANING, not surface words. Examples of correct reasoning:

  ✓ "the app keeps crashing"           → topic = app_performance  (crash = app problem)
  ✓ "I was charged twice"              → topic = billing/payment   (financial anomaly)
  ✓ "someone logged into my account"  → topic = security          (unauthorized access)
  ✓ "video won't load on the platform"→ topic = app_performance  (technical issue)
  ✗ Do NOT default to "general" unless no other topic truly fits

Available topics for {sector}:
{taxonomy["topics"]}

──── SEVERITY ────
{taxonomy["severity_notes"]}

──── URGENCY ────
  low    – can wait days; informational or minor
  medium – should be handled within a few hours
  high   – needs immediate attention; blocking, financial, or safety-related

──── CONFIDENCE ────
Your classification confidence from 0.0 to 1.0.
Lower confidence if feedback is: very short, sarcastic, multi-topic, ambiguous, emoji-only.

──── OUTPUT ────
Return ONLY valid JSON — no markdown, no fences, no extra keys:
{{
  "sentiment": "positive | neutral | negative",
  "intent": "<one intent from the list above>",
  "topic": "<one topic from the list above>",
  "urgency": "low | medium | high",
  "severity": "low | medium | high",
  "confidence": 0.85,
  "input_type": "text | emoji | mixed | rating",
  "emoji_detected": false,
  "needs_human_review": false,
  "review_reason": "",
  "human_readable": "One clear English sentence stating exactly what the customer means and feels.",
  "sector": "{sector}"
}}

──── RULES ────
  • Set needs_human_review=true if severity=high OR confidence < 0.6
  • human_readable must be faithful — preserve urgency, amounts, and facts; do not soften
  • For emoji-only input, infer meaning and set emoji_detected=true
  • Pick topic by the ROOT CAUSE, not the surface words used
  • If the customer mentions the app/platform behaving wrongly → topic is app_performance
  • If the customer mentions money/charges/refunds → topic is billing/payment/fees_billing
  • If the customer mentions account access/login/security → topic is account or security"""


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def understand(cleaned_feedback: str, sector: str | None = None) -> dict:
    """
    Analyze cleaned feedback and return structured understanding dict.

    Args:
        cleaned_feedback: English-normalized feedback text (from ingestion_agent)
        sector: Optional override. Falls back to config.SECTOR, then 'fintech'.

    Returns:
        dict with sentiment, intent, topic, urgency, severity, confidence,
        needs_human_review, review_reason, human_readable, sector, etc.
    """
    effective_sector = (sector or SECTOR or _DEFAULT_SECTOR).lower().strip()
    if effective_sector not in _SECTOR_TAXONOMY:
        print(f"[Understanding] Unknown sector '{effective_sector}', defaulting to '{_DEFAULT_SECTOR}'")
        effective_sector = _DEFAULT_SECTOR

    system_prompt = _build_system_prompt(effective_sector)

    from langchain_core.messages import HumanMessage, SystemMessage
    result = _llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=cleaned_feedback),
    ])
    text = re.sub(r"```json|```", "", result.content.strip()).strip()

    try:
        parsed = json.loads(text)
        parsed.setdefault("sector", effective_sector)
        return parsed
    except json.JSONDecodeError:
        return {
            "sentiment": "neutral",
            "intent": "general",
            "topic": "general",
            "urgency": "low",
            "severity": "low",
            "confidence": 0.5,
            "input_type": "text",
            "emoji_detected": False,
            "needs_human_review": False,
            "review_reason": "",
            "human_readable": cleaned_feedback,
            "sector": effective_sector,
        }