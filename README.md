# 📚 RAG Document Q&A Assistant

A Retrieval-Augmented Generation (RAG) based Document Question Answering system that allows users to upload a PDF and ask questions about its content.

The system retrieves relevant information from the uploaded document and uses an LLM to generate answers grounded in the document.

---

## 🚀 Project Overview

Traditional question-answering systems may generate answers from general knowledge, which can lead to irrelevant or unsupported responses.

This project uses **Retrieval-Augmented Generation (RAG)** to:

1. Load a PDF document
2. Split the document into smaller chunks
3. Convert chunks into vector embeddings
4. Retrieve relevant document sections
5. Combine vector search with keyword-based BM25 retrieval
6. Use Reciprocal Rank Fusion (RRF) to combine retrieval results
7. Generate an answer using Llama 3.2
8. Display the source pages used for the answer

---

## ✨ Features

- 📄 PDF document upload
- 🔎 Semantic vector search
- 🔤 BM25 keyword search
- 🔀 Reciprocal Rank Fusion (RRF)
- 🤖 Llama 3.2 3B for answer generation
- 🧠 Sentence Transformer embeddings
- 📑 Source page identification
- 🔍 Retrieved source snippets
- 💬 Conversation history
- 🗑️ Clear chat functionality
- 🖥️ Streamlit web interface

---

## 🏗️ System Architecture

```text
                PDF Document
                     │
                     ▼
              PDF Text Extraction
                     │
                     ▼
               Text Chunking
              Chunk Size: 1000
              Overlap: 200
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
   Vector Embeddings          BM25
   all-MiniLM-L6-v2       Keyword Search
          │                     │
          └──────────┬──────────┘
                     ▼
             Reciprocal Rank
                Fusion
                     │
                     ▼
              Relevant Chunks
                     │
                     ▼
              Llama 3.2 3B
                     │
                     ▼
              Final Answer
                     │
                     ▼
          Source Pages + Snippets