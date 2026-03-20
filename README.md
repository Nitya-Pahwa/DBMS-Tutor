# DBMS Concept Graph Tutor
An **AI-powered DBMS learning assistant** that combines:
- Retrieval-Augmented Generation (RAG)
- Concept Graph Visualization
- Multi-session Chat Interface

Built using **LangGraph + FAISS + Streamlit + Groq LLM**

## Features

- Ask any DBMS question  
- Context-aware answers using RAG  
- Source citations from PDFs  
- Automatic question classification  
- Concept graph visualization (for theory questions)  
- Multi-session chat history  

## Architecture

Pipeline:

User Query  
-> Classifier  
-> Retriever (FAISS)  
-> Reasoning (LLM)  
-> Answer Generator  

## Concept Graph

For questions like:
- Definitions
- Comparisons
- Relationships  

The system generates a **knowledge graph**:
- Nodes = DBMS concepts  
- Edges = relationships  

## Tech Stack

- **LLM**: Groq (LLaMA 3.1)
- **Framework**: LangGraph
- **Vector DB**: FAISS
- **Embeddings**: Sentence Transformers
- **Frontend**: Streamlit
- **Visualization**: PyVis

## Project Structure  
```bash
Nodes/  
├── classify.py  
├── retrieve.py  
├── reason.py  
├── generate.py  
└── safe_llm.py  
app.py  
graph_flow.py  
preprocess_index.py  
requirements.txt  
```
## Setup Instructions

1. Clone repo
```bash
[git clone https://github.com/your-username/dbms-concept-tutor.git
cd dbms-concept-tutor](https://github.com/Nitya-Pahwa/DBMS-Tutor.git)
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Add environment variables
```bash
GROQ_API_KEY=your_api_key_here
```

4. Build vector database
```bash
python preprocess_index.py
```

5. Run app
```bash
streamlit run app.py
```

## Dataset

DBMS PDFs (books, notes, slides)  
Converted into embeddings for retrieval       

## Glimpses of the Project  
<img width="1920" height="890" alt="Screenshot (1574)" src="https://github.com/user-attachments/assets/ffc22e4e-c5ee-46f4-a364-f0a57416c11e" /><br>

<img width="1920" height="912" alt="Screenshot (1575)" src="https://github.com/user-attachments/assets/0189385c-b4af-45dd-b176-eca99bcd0d9d" /><br>

<img width="1894" height="903" alt="Screenshot (1576)" src="https://github.com/user-attachments/assets/35e86164-13d0-4ff6-b6ee-071b90ada616" /><br>

<img width="1920" height="910" alt="Screenshot (1577)" src="https://github.com/user-attachments/assets/4859bf43-5c98-4c91-afa9-ca1080145193" /><br>

<img width="1920" height="907" alt="Screenshot (1581)" src="https://github.com/user-attachments/assets/60d00446-4d6d-4634-9ab9-6db5e7652e72" /><br>
