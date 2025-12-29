# HSxTech Genie

## Overview
HSxTech Genie is an AI-powered conversational assistant designed to answer **HSxTech and Odoo ERP–related questions** using a hybrid **Knowledge Graph + Web Search** approach. It delivers **accurate, concise, and non-technical answers** through a real-time chat interface.

## Key Features
- LangGraph-based intelligent agent
- Neo4j knowledge graph semantic search
- Google fallback search via Tavily
- Streaming responses with Chainlit
- Tool-controlled RAG (no hallucinations)
- FastAPI backend integration

## Tech Stack
- LLM: OpenAI (GPT-4o-mini)
- Agent Framework: LangGraph
- UI: Chainlit
- Backend: FastAPI
- Knowledge Graph: Neo4j
- Embeddings: text-embedding-3-small
- Web Search: Tavily

## Project Structure
- `main.py` – FastAPI app and Chainlit mount  
- `chainlit_app.py` – Chat UI, sessions, concurrency control  
- `langgraph_app.py` – Agent logic, tools, and graph flow  
- `similarity.py` – Neo4j semantic similarity search  

## How It Works
1. User submits a query via Chainlit UI  
2. Agent first searches Neo4j using semantic similarity  
3. Falls back to Google search if internal data is insufficient  
4. Final answer is streamed in clean markdown  

## Environment Variables
```env
OPENAI_API_KEY=
NEO4J_URI=
NEO4J_USERNAME=
NEO4J_PASSWORD=
NEO4J_DATABASE=
TAVILY_API_KEY=
```

## Run the Application
uvicorn main:app --reload


## Access the chatbot:

http://localhost:8000/chatbot
