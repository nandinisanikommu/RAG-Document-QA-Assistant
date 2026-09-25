# 📚 Hybrid RAG Document Q&A Assistant

A **Retrieval-Augmented Generation (RAG)** based Document Question Answering Assistant that allows users to upload a PDF and ask questions about its content.

The system combines **dense vector retrieval, BM25 keyword retrieval, Reciprocal Rank Fusion (RRF), Cross-Encoder reranking, and a local Llama 3.2 model** to generate grounded answers from the uploaded document.

---

## 🚀 Features

- 📄 Upload and process PDF documents
- 🔎 Dense vector search using `all-MiniLM-L6-v2`
- 🔤 Sparse keyword retrieval using **BM25Okapi**
- 🔀 Hybrid retrieval using **Reciprocal Rank Fusion (RRF)**
- 🎯 Cross-Encoder reranking using `ms-marco-MiniLM-L-6-v2`
- 🤖 Local answer generation using **Llama 3.2 3B**
- 📑 Page-level source references
- 🔎 Retrieved context snippets
- 🛡️ Document-grounded answers
- ❌ Explicit fallback when an answer cannot be found in the document
- 💻 Interactive Streamlit interface
- 🔒 Local document processing and local LLM inference

---

## 🏗️ System Architecture

```text
                  ┌──────────────────────┐
                  │      PDF Upload      │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │    PDF Extraction    │
                  │      PyPDFLoader      │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  Text Chunking       │
                  │  Chunk = 1000        │
                  │  Overlap = 200       │
                  └──────────┬───────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
      ┌──────────────────┐      ┌──────────────────┐
      │ Dense Retrieval  │      │ Sparse Retrieval │
      │ ChromaDB         │      │ BM25Okapi        │
      │ MiniLM Embedding │      │ Keyword Search   │
      └────────┬─────────┘      └────────┬─────────┘
               │                         │
               └────────────┬────────────┘
                            ▼
                  ┌──────────────────────┐
                  │ Reciprocal Rank      │
                  │ Fusion (RRF)         │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Cross-Encoder        │
                  │ Reranking             │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Relevant Context     │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Llama 3.2 3B         │
                  │ Ollama                │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Grounded Answer      │
                  │ + Source Pages       │
                  └──────────────────────┘