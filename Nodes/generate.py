"""
Module: generate.py

Purpose:
Formats the final answer and attaches citations.

Why:
Provides clean output and traceability of sources.

Input:
state["reasoned_text"], state["retrieved_docs"]

Output:
state["final_answer"]
"""

def generate(state):
    reasoned = state.get("reasoned_text")

    if not reasoned:
        reasoned = "No answer generated."

    docs = state.get("retrieved_docs", [])

    citations = sorted({
        d.metadata.get("source", "unknown")
        for d in docs
    })

    citation_text = "\n".join(f"- {c}" for c in citations) if citations else "No sources found."

    state["final_answer"] = f"""
**Answer:**

{reasoned}

**Sources used:**
{citation_text}
""".strip()

    return state
