# RAG-LLM

A modular Retrieval-Augmented Generation (RAG) framework built with Python. The project is designed with clean architecture principles, making each component independent and easily replaceable. It supports document ingestion, text chunking, embedding generation, FAISS vector storage, retrieval, prompt construction, and answer generation using a Large Language Model.

---

# Features

* Modular RAG pipeline
* PDF document ingestion
* Configurable text chunking
* Embedding generation
* FAISS vector database
* Similarity-based document retrieval
* Prompt template builder
* LLM-based answer generation
* YAML-based configuration
* Logging support
* Unit tests for individual modules

---

# Project Structure

```text
.
├── configs/
│   └── config.yaml          # Project configuration
├── data/
│   ├── doc.index            # FAISS index
│   └── doc.pkl              # Stored metadata
├── docs/                    # Input PDF documents
├── logs/
│   └── rag-llm.log
├── src/                     # Source code
├── test/                    # Unit tests
├── main.py                  # Entry point
└── README.md
```

---

# Architecture

The project follows a modular RAG pipeline:

```
PDF Documents
      │
      ▼
Data Ingestion
      │
      ▼
Chunking
      │
      ▼
Embedding
      │
      ▼
FAISS Vector Store
      │
      ▼
Retriever
      │
      ▼
Prompt Builder
      │
      ▼
LLM Generator
      │
      ▼
Final Answer
```

Each stage is implemented as an independent module, allowing different implementations to be plugged into the pipeline with minimal changes.

---

# Installation

Clone the repository

```bash
git clone https://github.com/Abolmw4/RAG-LLM.git

cd rag-llm
```

Create a virtual environment

```bash
python -m venv <your-env>
```

Activate it

Linux/macOS

```bash
source <your-env>/bin/activate
```

Windows

```bash
<your-env>\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Configuration

Project settings are stored in

```text
configs/config.yaml
```

Typical configurable parameters include

* Embedding model
* Chunk size
* Chunk overlap
* Vector database path
* LLM model
* Prompt template
* Retrieval parameters

---

# Usage

Run the project

```bash
python main.py
```

The pipeline performs the following steps:

1. Load PDF documents.
2. Split documents into chunks.
3. Generate embeddings.
4. Store embeddings in FAISS.
5. Retrieve relevant chunks.
6. Build a prompt.
7. Generate the final response using the LLM.

---

# Testing

Run all unit tests

```bash
python -m unittest discover test
```

or

```bash
pytest
```

---

# Technologies

* Python
* LangChain
* FAISS
* Ollama
* Sentence Transformers
* PyYAML
* unittest

---

# Future Work

* Hybrid Retrieval (BM25 + Dense Retrieval)
* Re-ranking
* Metadata filtering
* Multi-query retrieval
* Streaming responses
* Web interface
* REST API
* Support for additional document formats
* Evaluation pipeline
* Docker deployment

