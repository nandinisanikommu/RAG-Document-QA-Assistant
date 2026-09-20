# 📚 Hybrid RAG Document QA Assistant

A privacy-preserving, local Document Question-Answering system built using **LangChain**, **ChromaDB**, **BM25**, and **Ollama (Llama 3.2:3B)**.

This project features a **two-stage hybrid search architecture** (Dense vector search + Sparse keyword search) enhanced by **Reciprocal Rank Fusion (RRF)** and **Cross-Encoder reranking** to eliminate LLM hallucinations and deliver precise, page-cited answers from PDF documents.

---

## 🚀 Features

- **🔒 100% Local & Private Execution:** Powered by Ollama (`llama3.2:3b`), ensuring zero external data leakage or cloud API costs.
- **🔍 Two-Stage Hybrid Retrieval:**
  - **Dense Search:** Semantic vector similarity matching via ChromaDB (`all-MiniLM-L6-v2`).
  - **Sparse Search:** Exact keyword and domain-term matching via `BM25Okapi`.
- **🔀 Reciprocal Rank Fusion (RRF):** Merges and normalizes rank scores from both dense and sparse retrievers.
- **🎯 Cross-Encoder Reranking:** Re-evaluates top-k passages using `ms-marco-MiniLM-L-6-v2` to pass only high-confidence context to the LLM.
- **📌 Page-Level Citations:** Displays exact source page numbers alongside retrieved text snippets for complete output verification.
- **💬 Interactive Streamlit UI:** Features conversational chat history, context expansion toggles, and real-time streaming response generation.

---

## 🏗️ System Architecture

```text
┌─────────────────┐
│  Uploaded PDF   │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Text Chunking   │ (RecursiveCharacterTextSplitter)
└────────┬────────┘
         ├──────────────────────────────────────┐
         ▼                                      ▼
┌─────────────────┐                    ┌─────────────────┐
│  ChromaDB Vector│                    │  BM25 Index     │
│  (Dense Search) │                    │ (Sparse Search) │
└────────┬────────┘                    └────────┬────────┘
         │                                      │
         └──────────────────┬───────────────────┘
                            ▼
               ┌──────────────────────────┐
               │ Reciprocal Rank Fusion   │ (RRF Scoring)
               └────────────┬─────────────┘
                            ▼
               ┌──────────────────────────┐
               │  Cross-Encoder Reranker  │ (ms-marco-MiniLM-L-6-v2)
               └────────────┬─────────────┘
                            ▼
               ┌──────────────────────────┐
               │ Local Ollama LLM         │ (Llama 3.2:3B)
               └────────────┬─────────────┘
                            ▼
               ┌──────────────────────────┐
               │ Streamlit UI Answer      │ (With Page Citations)
               └──────────────────────────┘