"""
Module: graph_flow.py

Purpose:
Defines LangGraph pipeline for DBMS tutor.

Flow:
classify → retrieve → reason → generate

Why:
Ensures modular and structured execution of RAG pipeline.
"""

from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from Nodes.classify import classify_query
from Nodes.retrieve import retrieve
from Nodes.reason import reason
from Nodes.generate import generate

class StateSchema(TypedDict, total=False):
    query: str
    question_type: str
    retrieved_docs: List
    context: str
    reasoned_text: str
    final_answer: str

# Builds and compiles LangGraph pipeline
def compile_graph():

    # Initialize graph with state schema
    graph = StateGraph(StateSchema)

    # Add pipeline nodes
    graph.add_node("classify", classify_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("reason", reason)
    graph.add_node("generate", generate)

    # Define starting node
    graph.set_entry_point("classify")

    # Define execution flow
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "reason")
    graph.add_edge("reason", "generate")
    graph.add_edge("generate", END)

    return graph.compile()
