from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from config import MODEL_NAME

llm = ChatGroq(model=MODEL_NAME, temperature=0.4)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a professional customer support agent. 
Use the following context from our knowledge base to write a helpful, empathetic reply.
Be concise, friendly, and solution-focused.

Context:
{context}"""),
    ("human", "Customer feedback: {feedback}\nAnalysis: {analysis}")
])

def generate_response(feedback: str, analysis: dict, context: str) -> str:
    chain = prompt | llm
    result = chain.invoke({
        "feedback": feedback,
        "analysis": str(analysis),
        "context": context
    })
    return result.content
