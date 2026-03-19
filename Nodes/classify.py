"""
Module: classify.py

Purpose:
Classifies the user query into a DBMS question type 
(e.g., definition, comparison, relationship, process, sql).

Why:
Helps downstream modules adapt response style and decide 
whether to generate a concept graph.

Input:
state["query"] -> user question

Output:
state["question_type"] -> classified category
"""

from Nodes.safe_llm import safe_invoke

def classify_query(state):
    # Get user query from state
    query = state.get("query", "")

    # Prompt LLM to classify question
    prompt = f"""
Classify the DBMS question into:
definition, comparison, relationship, process, sql

Question:
{query}

Return only the category name.
"""
    # Call LLM and normalize response
    raw = safe_invoke(prompt).lower()
 
    # Map LLM output to predefined categories
    if "definition" in raw:
        qtype = "definition"
    elif "compare" in raw:
        qtype = "comparison"
    elif "relation" in raw:
        qtype = "relationship"
    elif "process" in raw or "how" in raw:
        qtype = "process"
    elif "sql" in raw:
        qtype = "sql"
    else:
        qtype = "definition"

    # Store classification result in state
    state["question_type"] = qtype
    return state
