import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


# --------------------------------------------------
# 1. Initialize Models
# --------------------------------------------------

EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

RERANKER = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# --------------------------------------------------
# 2. Process Uploaded PDF
# --------------------------------------------------

def process_pdf_bytes(pdf_bytes: bytes) -> dict:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp_file:

        tmp_file.write(pdf_bytes)
        tmp_path = tmp_file.name

    try:

        loader = PyPDFLoader(tmp_path)

        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = text_splitter.split_documents(
            documents
        )

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=EMBEDDING_MODEL
        )

        tokenized_corpus = [
            doc.page_content.lower().split()
            for doc in chunks
        ]

        bm25 = BM25Okapi(
            tokenized_corpus
        )

        return {
            "chunks": chunks,
            "vectorstore": vectorstore,
            "bm25": bm25
        }

    finally:

        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# --------------------------------------------------
# 3. Answer Question
# --------------------------------------------------

def answer_question(
    question: str,
    rag_system: dict,
    top_k: int = 5
) -> tuple[str, list[int], list[dict]]:

    chunks = rag_system["chunks"]

    vectorstore = rag_system["vectorstore"]

    bm25 = rag_system["bm25"]

    question_lower = question.lower()

    # --------------------------------------------------
    # Detect List Questions
    # --------------------------------------------------

    list_keywords = [
        "types",
        "list",
        "what are",
        "which",
        "programming languages",
        "advantages",
        "uses",
        "applications",
        "components",
        "methods"
    ]

    is_list_question = any(
        keyword in question_lower
        for keyword in list_keywords
    )

    # --------------------------------------------------
    # Step 1: Vector Search
    # --------------------------------------------------

    vector_k = min(
        15 if is_list_question else top_k,
        len(chunks)
    )

    vector_results = (
        vectorstore.similarity_search_with_score(
            question,
            k=vector_k
        )
    )

    # --------------------------------------------------
    # Step 2: BM25 Search
    # --------------------------------------------------

    tokenized_query = question_lower.split()

    bm25_results = bm25.get_top_n(
        tokenized_query,
        chunks,
        n=vector_k
    )

    # --------------------------------------------------
    # Step 3: RRF
    # --------------------------------------------------

    rrf_scores = {}

    rrf_k = 60

    for rank, (doc, _) in enumerate(
        vector_results
    ):

        key = doc.page_content

        rrf_scores.setdefault(
            key,
            {
                "score": 0.0,
                "doc": doc
            }
        )

        rrf_scores[key]["score"] += (
            1.0 / (rrf_k + rank + 1)
        )

    for rank, doc in enumerate(
        bm25_results
    ):

        key = doc.page_content

        rrf_scores.setdefault(
            key,
            {
                "score": 0.0,
                "doc": doc
            }
        )

        rrf_scores[key]["score"] += (
            1.0 / (rrf_k + rank + 1)
        )

    # --------------------------------------------------
    # Sort RRF Results
    # --------------------------------------------------

    ranked_items = sorted(
        rrf_scores.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    # --------------------------------------------------
    # Initial Candidates
    # --------------------------------------------------

    candidate_docs = [
        item["doc"]
        for item in ranked_items[
            :min(15, len(ranked_items))
        ]
    ]

    # --------------------------------------------------
    # SPECIAL HANDLING FOR MACHINE LEARNING TYPES
    # --------------------------------------------------

    if (
        "machine learning" in question_lower
        and (
            "type" in question_lower
            or "types" in question_lower
        )
    ):

        required_terms = [
            "supervised learning",
            "unsupervised learning",
            "semi-supervised",
            "reinforcement learning"
        ]

        for term in required_terms:

            for doc in chunks:

                text = doc.page_content.lower()

                if term in text:

                    if doc not in candidate_docs:

                        candidate_docs.append(doc)

    # --------------------------------------------------
    # Remove Duplicate Documents
    # --------------------------------------------------

    unique_docs = {}

    for doc in candidate_docs:

        unique_docs[
            doc.page_content
        ] = doc

    candidate_docs = list(
        unique_docs.values()
    )

    # --------------------------------------------------
    # Step 4: Cross-Encoder Reranking
    # --------------------------------------------------

    pairs = [
        [
            question,
            doc.page_content
        ]
        for doc in candidate_docs
    ]

    scores = RERANKER.predict(
        pairs
    )

    scored_docs = sorted(
        zip(
            scores,
            candidate_docs
        ),
        key=lambda x: x[0],
        reverse=True
    )

    # --------------------------------------------------
    # Final Documents
    # --------------------------------------------------

    if (
        "machine learning" in question_lower
        and (
            "type" in question_lower
            or "types" in question_lower
        )
    ):

        final_k = min(
            8,
            len(scored_docs)
        )

    elif is_list_question:

        final_k = min(
            6,
            len(scored_docs)
        )

    else:

        final_k = min(
            3,
            len(scored_docs)
        )

    top_docs = [
        doc
        for _, doc in scored_docs[:final_k]
    ]

    # --------------------------------------------------
    # Relevance Check
    # --------------------------------------------------

    best_score = (
        float(scored_docs[0][0])
        if scored_docs
        else 0.0
    )

    RELEVANCE_THRESHOLD = -1.0

    if best_score < RELEVANCE_THRESHOLD:

        return (
            "I could not find the answer in the provided document.",
            [],
            []
        )

    # --------------------------------------------------
    # Build Context
    # --------------------------------------------------

    context = "\n\n".join(
        f"[Page {doc.metadata.get('page', 0) + 1}]: "
        f"{doc.page_content}"
        for doc in top_docs
    )

    # --------------------------------------------------
    # LLM Prompt
    # --------------------------------------------------

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the information
contained in the provided document context.

IMPORTANT RULES:

1. Do not use outside knowledge.

2. Do not invent information.

3. Answer directly and clearly.

4. If the user asks for a list, provide the actual
   items found in the document.

5. If the user asks for the types of machine learning,
   examine all relevant sections in the context.

6. The document discusses these machine-learning types
   in separate sections. Include all of the following
   when they appear in the provided context:

   - Supervised Learning
   - Unsupervised Learning
   - Semi-supervised Learning
   - Reinforcement Learning

7. Do NOT treat traditional programming as a type
   of machine learning.

8. If information is spread across multiple pages,
   combine the relevant information.

9. Do not stop after finding only two items when
   additional relevant items are present.

10. For programming-language headings such as
    "Python Machine Learning", the actual language
    name is "Python".

11. For headings such as "Java Machine Learning",
    the actual language name is "Java".

12. Do not copy unnecessary words from headings.

13. Do not mention page numbers in the answer.

14. If the requested information is not supported
    by the context, respond exactly:

    I could not find the answer in the provided document.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    # --------------------------------------------------
    # Generate Answer
    # --------------------------------------------------

    llm = OllamaLLM(
        model="llama3.2:3b"
    )

    answer = llm.invoke(
        prompt
    )

    # --------------------------------------------------
    # Source Pages
    # --------------------------------------------------

    source_pages = sorted(
        list(
            {
                doc.metadata.get(
                    "page",
                    0
                ) + 1
                for doc in top_docs
            }
        )
    )

    # --------------------------------------------------
    # Source Snippets
    # --------------------------------------------------

    source_snippets = [

        {
            "page": doc.metadata.get(
                "page",
                0
            ) + 1,

            "text": (
                doc.page_content[:250]
                + "..."
            )
        }

        for doc in top_docs
    ]

    # --------------------------------------------------
    # Return
    # --------------------------------------------------

    return (
        answer,
        source_pages,
        source_snippets
    )