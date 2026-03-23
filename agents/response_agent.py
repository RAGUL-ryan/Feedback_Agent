from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from config import MODEL_NAME

llm = ChatGroq(model=MODEL_NAME, temperature=0.4)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a professional, empathetic customer support agent.
Use the context from our knowledge base to write a helpful reply.
The feedback may have been submitted as emojis, star ratings, or text — respond naturally to the meaning.
Be warm, concise, and solution-focused.

Context:
{context}"""),
    ("human", """Original input: {original_input}
Interpreted as: {human_readable}
Analysis: {analysis}

Write a warm, helpful reply addressing what the customer means:""")
])

def generate_response(feedback: str, analysis: dict, context: str, original_input: str = "", human_readable: str = "") -> str:
    chain = prompt | llm
    result = chain.invoke({
        "feedback": feedback,
        "original_input": original_input or feedback,
        "human_readable": human_readable or feedback,
        "analysis": str(analysis),
        "context": context
    })
    return result.content