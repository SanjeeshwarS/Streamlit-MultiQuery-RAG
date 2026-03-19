# 🤖 Sanjii's RAG System: Beyond Basic Similarity Search

Most RAG (Retrieval-Augmented Generation) setups fail when a user's question doesn't perfectly match the document's text. I built this system to solve that using **Multi-Query Retrieval**. 

By rephrasing a single user prompt into 5 different semantic perspectives, this assistant "digs deeper" into your PDFs to find the right context—even if the phrasing is vague or complex.

---

## 🧐 Why I Created This
The main goal was to move away from "Black Box" AI and build a tool that was:
1. **Private:** 100% local execution. No data ever leaves my machine.
2. **Smart:** Uses Multi-Query logic to understand intent, not just keywords.
3. **Transparent:** A "Source Documents" feature shows you the exact evidence used for every answer.

---

## ✨ Key Features
* **Multi-Query Synthesis:** Automatically generates 5 diverse search perspectives to overcome similarity search limitations.
* **Local-First Privacy:** Powered by **Ollama (Llama 3.2)** and **ChromaDB** for secure, local-first data processing.
* **Intelligent Vector Retrieval:** Optimized chunking using Recursive Character Splitting for better context retention.
* **Real-time Status Dashboard:** A custom Streamlit sidebar that tracks Ollama's status and the health of the Vector Database.
* **Source Transparency:** Built-in expanders to verify exactly which chunks of the PDF the AI is reading.

---

## 📂 Project Structure
Below is the directory layout for the system. I have organized it to separate the raw data, the vector database, and the application logic:

```text
Sanjii-RAG-System/
├── data/               # Input folder for your PDF documents
│   └── Galaxies.pdf    # Default knowledge base file
├── my_chroma_db/       # Persistent Vector Database (automatically managed by Chroma)
│   ├── chroma.sqlite3  # Database metadata
│   └── [hash-folders]/ # Compressed vector embeddings
├── RagStreamlit.py     # Core Application: UI, RAG Chain, and Retrieval logic
├── requirements.txt    # Python dependencies (LangChain, Streamlit, etc.)
└── .gitignore          # Prevents heavy DB and local data from being pushed to Git

## 🛠️ Tech Stack
* **LLM:** [Ollama](https://ollama.com/) (Llama 3.2)
* **Embeddings:** `nomic-embed-text`
* **Orchestration:** LangChain
* **Vector Store:** ChromaDB
* **Frontend:** Streamlit

---

## 🧠 How it Works (The Multi-Query Edge)

Standard RAG systems often fail if the user's question doesn't perfectly match the wording in the document. This system solves that by:
1. **Generating Perspectives:** The LLM creates 5 variations of the input query.
2. **Vector Search:** All 5 queries are used to search the **ChromaDB** vector store.
3. **Context Fusion:** The system retrieves the most relevant chunks from all queries, providing a richer context for the final answer.



---

## 🚀 Getting Started

### 1. Prerequisites
* Install [Ollama](https://ollama.com/)
* Pull the required models:
  ```bash
  ollama pull llama3.2
  ollama pull nomic-embed-text