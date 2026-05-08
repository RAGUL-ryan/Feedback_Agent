"""
agents/response_agent.py
─────────────────────────────────────────────────────────────────────────────
Customer response generation agent.

v3 changes:
  • Sector-aware system prompt — tone and content rules adapt per domain:
    fintech, education, healthcare, food, ecommerce
  • generate_response() accepts optional `sector` param
  • 3-sentence structure preserved from v2
"""

from langchain_core.prompts import ChatPromptTemplate
#from langchain_groq import ChatGroq
from config import MODEL_NAME, SECTOR

#_llm = ChatGroq(model=MODEL_NAME, temperature=0.4)
from langchain_openai import ChatOpenAI
_llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
# Sector-specific tone and content guidance injected into the system prompt
_SECTOR_GUIDANCE = {
    "fintech": (
        "You work for a fintech / banking company. "
        "Be precise about financial terms. Never guess at amounts, timelines, or policy details. "
        "For security/fraud issues, emphasize that the team will investigate immediately."
    ),
    "education": (
        "You work for an online education platform. "
        "Be encouraging and supportive in tone. "
        "For access or billing issues, reassure the student that their learning won't be disrupted."
    ),
    "healthcare": (
        "You work for a healthcare / hospital platform. "
        "Be calm, empathetic, and professional. Patient wellbeing is the top priority. "
        "Never provide medical advice. For billing disputes, explain the process clearly."
    ),
    "food": (
        "You work for a food delivery platform. "
        "Be warm and apologetic for quality or delivery issues. "
        "For food safety concerns, escalate urgency and assure immediate action."
    ),
    "ecommerce": (
        "You work for an ecommerce platform. "
        "Be solution-focused — returns, refunds, and replacements are standard. "
        "Give clear timelines and process steps when available."
    ),
}

_SYSTEM_TEMPLATE = """{sector_guidance}

Write a reply using this exact 3-sentence structure:
  Sentence 1: Acknowledge — show you understood what they said and how they feel.
  Sentence 2: Answer — give the direct, useful answer using the knowledge base context below.
  Sentence 3: Next step — one clear action for the customer to take.

Tone rules by intent:
  complaint / bug_report  → acknowledge the problem first, then solve it
  praise                  → thank them warmly and briefly — do not over-explain
  question                → give the direct answer; skip acknowledgment if it wastes space
  suggestion              → thank them and explain how feedback is used
  refund_request          → empathize, give process steps, set clear expectations
  fraud_report            → acknowledge urgency, assure investigation, give next step
  order_issue             → apologize, give resolution path, set timeline expectation
  general                 → be warm and helpful

Content rules:
  • Use the knowledge base context to give accurate, specific answers.
  • If context does not answer the question, say so and offer a next step (e.g. contact support).
  • Never fabricate policy details, amounts, or timelines.
  • Do NOT reveal internal scores, edge case flags, or analysis details to the customer.
  • Respond to the MEANING of the feedback — not the raw emoji or rating symbols.
  • Keep it under 60 words total.

Knowledge base context:
{context}"""

_HUMAN = """Customer input: {original_input}
What they mean: {human_readable}
Intent: {intent} | Topic: {topic} | Urgency: {urgency}

Write a 3-sentence reply:"""


def generate_response(
    feedback: str,
    analysis: dict,
    context: str,
    original_input: str = "",
    human_readable: str = "",
    sector: str | None = None,
) -> str:
    effective_sector = (sector or analysis.get("sector") or SECTOR or "fintech").lower()
    sector_guidance = _SECTOR_GUIDANCE.get(effective_sector, _SECTOR_GUIDANCE["fintech"])

    system_msg = _SYSTEM_TEMPLATE.format(
        sector_guidance=sector_guidance,
        context=context or "No specific knowledge base context retrieved.",
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", _HUMAN),
    ])

    result = (prompt | _llm).invoke({
        "original_input": original_input or feedback,
        "human_readable": human_readable or feedback,
        "intent": analysis.get("intent", "general"),
        "topic": analysis.get("topic", "general"),
        "urgency": analysis.get("urgency", "low"),
    })
    return result.content.strip()