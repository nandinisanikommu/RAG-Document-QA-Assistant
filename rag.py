import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# 1. Initialize models globally
EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Standard Hugging Face Cross-Encoder model
RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def process_pdf_bytes(pdf_bytes: bytes) -> dict:
  with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
    tmp_file.write(pdf_bytes)
    tmp_path = tmp_file.name

  try:
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=EMBEDDING_MODEL
    )

    tokenized_corpus = [doc.page_content.lower().split() for doc in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    return {"chunks": chunks, "vectorstore": vectorstore, "bm25": bm25}

  finally:
    if os.path.exists(tmp_path):
      os.remove(tmp_path)


def answer_question(
    question: str, rag_system: dict
) -> tuple[str, list[int], list[dict]]:
  chunks = rag_system["chunks"]
  vectorstore = rag_system["vectorstore"]
  bm25 = rag_system["bm25"]

  k_retrieve = min(15, len(chunks))

  # Step 1: Dense Vector Search
  vector_results = vectorstore.similarity_search_with_score(
      question, k=k_retrieve
  )

  # Step 2: Sparse BM25 Keyword Search
  tokenized_query = question.lower().split()
  bm25_results = bm25.get_top_n(tokenized_query, chunks, n=k_retrieve)

  # Step 3: Reciprocal Rank Fusion (RRF)
  rrf_scores = {}
  rrf_k = 60

  for rank, (doc, _) in enumerate(vector_results):
    key = doc.page_content
    rrf_scores.setdefault(key, {"score": 0.0, "doc": doc})
    rrf_scores[key]["score"] += 1.0 / (rrf_k + rank + 1)

  for rank, doc in enumerate(bm25_results):
    key = doc.page_content
    rrf_scores.setdefault(key, {"score": 0.0, "doc": doc})
    rrf_scores[key]["score"] += 1.0 / (rrf_k + rank + 1)

  ranked_items = sorted(
      rrf_scores.values(), key=lambda x: x["score"], reverse=True
  )
  candidate_docs = [item["doc"] for item in ranked_items[:10]]

  # Step 4: Two-Stage Reranking via SentenceTransformers CrossEncoder
  pairs = [[question, doc.page_content] for doc in candidate_docs]
  scores = RERANKER.predict(pairs)

  # Sort candidate documents by cross-encoder score
  scored_docs = sorted(
      zip(scores, candidate_docs), key=lambda x: x[0], reverse=True
  )
  top_docs = [doc for _, doc in scored_docs[:3]]

  # Step 5: Context Assembly & Dynamic LLM Generation
  context = "\n\n".join(
      f"[Page {doc.metadata.get('page', 0) + 1}]: {doc.page_content}"
      for doc in top_docs
  )

  prompt = f"""You are a document QA assistant. Answer the user's question using ONLY the context provided below.
If the answer cannot be found in the context, state: "I could not find the answer in the provided document."

Context:
{context}

Question: {question}
Answer:"""

  llm = OllamaLLM(model="llama3.2:3b")
  answer = llm.invoke(prompt)

  source_pages = sorted(
      list({doc.metadata.get("page", 0) + 1 for doc in top_docs})
  )
  source_snippets = [
      {
          "page": doc.metadata.get("page", 0) + 1,
          "text": doc.page_content[:250] + "...",
      }
      for doc in top_docs
  ]

  return answer, source_pages, source_snippets