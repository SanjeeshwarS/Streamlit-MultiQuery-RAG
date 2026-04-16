# Multi-Query RAG System

A privacy-focused Retrieval-Augmented Generation (RAG) system utilizing multi-query retrieval for enhanced document context synthesis. This implementation runs entirely locally using Ollama and ChromaDB.

## Overview

Standard RAG systems often struggle with keyword-dependent similarity searches. This project implements an Advanced RAG approach using Multi-Query Retrieval. By generating multiple semantic versions of a user's question, the system retrieves a more comprehensive set of context from local vector stores.

## Technical Stack

| Category | Technologies Used |
| :--- | :--- |
| Language | Python |
| AI Orchestration | LangChain |
| Local Inference | Ollama (Llama 3.2), Nomic Embeddings |
| Data & Retrieval | ChromaDB, Multi-Query Retrieval |
| Frameworks | Streamlit |

## Key Features

*   **Multi-Query Retrieval:** Synthesizes multiple search perspectives to improve retrieval accuracy.
*   **Local Execution:** All data processing and inference are performed locally for maximum privacy.
*   **Context Transparency:** Built-in verification of source documents for every response.
*   **System Monitoring:** Real-time tracking of model connectivity and database health.

## Installation and Setup

### 1. Prerequisites
*   Ollama (Llama 3.2 and Nomic-embed-text models)
*   Python 3.10+

### 2. Dependencies
```bash
pip install -r requirements.txt
```

### 3. Usage
```bash
streamlit run RagStreamlit.py
```

## Project Structure

*   `RagStreamlit.py`: Streamlit interface and application logic.
*   `Rag.py`: Core RAG retrieval engine.
*   `data/`: Input directory for PDF documents.
*   `my_chroma_db/`: Local vector database storage.
