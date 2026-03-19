"""
Module: reason.py

Purpose:
Generates answer using LLM based on retrieved context.

Why:
Core reasoning step of RAG pipeline.

Input:
state["query"], state["context"]

Output:
state["reasoned_text"]
"""

from Nodes.safe_llm import safe_invoke

# Generates answer using LLM based on query + retrieved context
def reason(state):
    # Extract query
    query = state.get("query", "")

    # Extract retrieved context
    context = state.get("context", "")

    # Build prompt combining context and question
    prompt = f"""
Answer the DBMS question clearly.

Context:
{context}

Question:
{query}
"""
    # Call LLM to generate answer
    answer = safe_invoke(prompt)

    state["reasoned_text"] = answer
    return state
