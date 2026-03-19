"""
Script: preprocess_index.py

Purpose:
Creates FAISS vector database from PDF documents.

Steps:
1. Load PDFs
2. Split into chunks
3. Generate embeddings
4. Store in FAISS index

Run this BEFORE starting the app.
"""

import os
from pathlib import Path
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import dotenv

dotenv.load_dotenv()

DATA_DIR = Path(r"C:\Users\Hp\Desktop\CS_Projects\Agentic AI\DBMS RAG Based Project\Dataset_Pdfs")
VECTORSTORE_PATH = Path(os.getenv("VECTORSTORE_PATH", "vectorstore/faiss_db"))
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Loads PDF documents and attaches metadata
def load_docs(data_dir):
    docs = []
    for p in tqdm(sorted(data_dir.glob("*"))):
        if p.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(p))
        else:
            print(f"Skipping unsupported file: {p.name}")
            continue

        loaded = loader.load()
        for d in loaded:
            d.metadata["source"] = p.name
        docs.extend(loaded)
    return docs

# Splits documents into smaller chunks for embeddings
def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(docs)

# Creates FAISS vector store from document chunks
def build_vectorstore(chunks, persist_path):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = FAISS.from_documents(chunks, embeddings)
    os.makedirs(persist_path.parent, exist_ok=True)
    db.save_local(str(persist_path))
    return db


def main():
    print("Loading documents (PDF/PPTX)...")
    docs = load_docs(DATA_DIR)
    print(f"Loaded {len(docs)} documents.")
    print("Splitting into chunks...")
    chunks = split_docs(docs)
    print(f"Total chunks: {len(chunks)}")
    print("Building FAISS vectorstore...")
    db = build_vectorstore(chunks, VECTORSTORE_PATH)
    print("Vectorstore saved to", VECTORSTORE_PATH)


if __name__ == "__main__":
    main()
