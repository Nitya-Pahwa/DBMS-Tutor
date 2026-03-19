"""
Module: retrieve.py

Purpose:
Retrieves relevant documents from FAISS vector database 
using semantic search.

Why:
Provides context to the LLM for accurate answers (RAG).

Input:
state["query"]

Output:
state["retrieved_docs"] -> top-k documents
state["context"] -> combined text for reasoning
"""

import os
import dotenv
dotenv.load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "vectorstore/faiss_db")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "4"))

# Global variables to cache vector DB and embeddings 
_db = None
_embeddings = None

# Loads FAISS vector database
def _load_db():
    global _db, _embeddings
    if _db is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        _db = FAISS.load_local(
            VECTORSTORE_PATH,
            _embeddings,
            allow_dangerous_deserialization=True
        )
    return _db

# Retrieves top-k relevant documents based on query
def retrieve(state):
    # Get user query
    query = state.get("query", "")
    db = _load_db()
    # Create retriever with top-k search
    retriever = db.as_retriever(search_kwargs={"k": TOP_K})
    # Perform similarity search
    docs = retriever.invoke(query)

    print(f"[DEBUG] Retrieved documents: {len(docs)}")

    # Save retrieved documents in state
    state["retrieved_docs"] = docs

    # Combine retrieved docs into a single context string
    state["context"] = "\n\n".join(
        f"[{d.metadata.get('source','unknown')}] {d.page_content}"
        for d in docs
    )
    return state
