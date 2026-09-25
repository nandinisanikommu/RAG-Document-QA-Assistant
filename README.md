# 📚 Hybrid RAG Document Q&A Assistant

A privacy-focused, local **Document Question-Answering system** built using **LangChain, ChromaDB, BM25, Cross-Encoder reranking, and Ollama (Llama 3.2:3B)**.

The system allows users to upload a PDF document and ask questions about its content. It combines **dense vector search** and **sparse keyword search** using **Reciprocal Rank Fusion (RRF)**, followed by **Cross-Encoder reranking**, to retrieve relevant document context before generating grounded answers.

The application also provides **page-level source references and retrieved context snippets**, making answers easier to verify against the original document.

---

## 🚀 Features

### 🔍 Hybrid Retrieval

Combines two complementary retrieval approaches:

- **Dense Vector Search**
  - Uses ChromaDB
  - Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
  - Captures semantic similarity between questions and document chunks

- **Sparse Keyword Search**
  - Uses `BM25Okapi`
  - Captures exact keywords and domain-specific terms

### 🔀 Reciprocal Rank Fusion (RRF)

Combines results from dense vector search and BM25 retrieval using **Reciprocal Rank Fusion** to produce a unified ranking of candidate document chunks.

### 🎯 Cross-Encoder Reranking

The retrieved candidates are reranked using:

`cross-encoder/ms-marco-MiniLM-L-6-v2`

This provides a second-stage relevance check before the final context is passed to the language model.

### 🤖 Local LLM

Uses:

`Llama 3.2:3B`

through **Ollama** for local answer generation.

The system does not require an external LLM API for inference.

### 📌 Page-Level Sources

Answers include the document pages used to generate the response.

Users can also expand the retrieved context snippets to inspect the supporting text.

### 💬 Interactive Streamlit Interface

The application provides:

- PDF upload
- Conversational chat history
- Dynamic document questions
- Retrieval settings
- Source page display
- Retrieved context expansion
- Clear chat functionality

### 🛡️ Grounded Answers

The LLM is instructed to answer using the retrieved document context and avoid introducing information that is not supported by the uploaded document.

For questions that cannot be answered from the document, the system can return:

> `I could not find the answer in the provided document.`

---

# 🏗️ System Architecture

```text
                 ┌────────────────────┐
                 │    Uploaded PDF    │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │    PDF Loading     │
                 │    PyPDFLoader     │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │    Text Chunking   │
                 │  Chunk Size: 1000  │
                 │  Overlap: 200      │
                 └─────────┬──────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
     ┌──────────────────┐      ┌──────────────────┐
     │    ChromaDB      │      │      BM25        │
     │  Dense Retrieval │      │ Sparse Retrieval │
     └────────┬─────────┘      └────────┬─────────┘
              │                         │
              └────────────┬────────────┘
                           ▼
                 ┌────────────────────┐
                 │ Reciprocal Rank    │
                 │ Fusion (RRF)       │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ Cross-Encoder      │
                 │ Reranking          │
                 │ ms-marco-MiniLM    │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │  Llama 3.2:3B      │
                 │      Ollama        │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │   Streamlit UI     │
                 │ Answer + Sources   │
                 └────────────────────┘