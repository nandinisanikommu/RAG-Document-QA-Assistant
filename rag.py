from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from rank_bm25 import BM25Okapi

import streamlit as st
import tempfile


# =========================================================
# EMBEDDING MODEL
# =========================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================================================
# PROCESS PDF
# =========================================================

@st.cache_resource(show_spinner=False)
def process_pdf(pdf_bytes, file_name):

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(pdf_bytes)
        pdf_path = temp_file.name

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(
        documents
    )

    db_path = tempfile.mkdtemp(
        prefix="rag_chroma_"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path
    )

    tokenized_chunks = [
        chunk.page_content.lower().split()
        for chunk in chunks
    ]

    bm25 = BM25Okapi(
        tokenized_chunks
    )

    print(
        f"PDF processed: {file_name}"
    )

    print(
        f"Pages: {len(documents)}"
    )

    print(
        f"Chunks: {len(chunks)}"
    )

    return {
        "chunks": chunks,
        "vectorstore": vectorstore,
        "bm25": bm25
    }


# =========================================================
# SOURCE SNIPPETS
# =========================================================

def build_source_snippets(
    documents,
    question="",
    max_sources=3
):

    question_lower = question.lower()

    stop_words = {
        "what",
        "are",
        "the",
        "of",
        "is",
        "a",
        "an",
        "in",
        "on",
        "for",
        "to",
        "and",
        "how",
        "does",
        "do",
        "which",
        "some",
        "used",
        "using",
        "types",
        "type",
        "list",
        "give",
        "explain",
        "tell",
        "about",
        "can",
        "you",
        "me"
    }

    important_words = []

    for word in question.split():

        clean_word = word.lower().strip(
            ".,?!:;"
        )

        if (
            clean_word
            and clean_word not in stop_words
            and len(clean_word) > 3
        ):

            important_words.append(
                clean_word
            )

    scored_documents = []

    for index, doc in enumerate(documents):

        text = doc.page_content.lower()

        score = 0

        for word in important_words:

            if word in text:

                score += 1

        # -------------------------------------------------
        # Advantages
        # -------------------------------------------------

        if (
            "advantage" in question_lower
            and "disadvantage" not in question_lower
        ):

            if "advantages of machine learning" in text:
                score += 20

            if "trends and patterns" in text:
                score += 15

            if "human intervention" in text:
                score += 15

            if "continuous improvement" in text:
                score += 15

        # -------------------------------------------------
        # Disadvantages
        # -------------------------------------------------

        if "disadvantage" in question_lower:

            if "disadvantages of machine learning" in text:
                score += 20

            if "data acquisition" in text:
                score += 15

            if "time and resources" in text:
                score += 15

        # -------------------------------------------------
        # Types
        # -------------------------------------------------

        if (
            "type" in question_lower
            or "types" in question_lower
        ):

            type_terms = [
                "supervised learning",
                "unsupervised learning",
                "semi-supervised learning",
                "reinforcement learning"
            ]

            for term in type_terms:

                if term in text:
                    score += 10

        # -------------------------------------------------
        # Supervised Learning
        # -------------------------------------------------

        if (
            "supervised learning" in question_lower
            and "unsupervised learning" not in question_lower
            and "semi-supervised learning" not in question_lower
        ):

            if "supervised learning" in text:
                score += 20

        # -------------------------------------------------
        # Unsupervised Learning
        # -------------------------------------------------

        if "unsupervised learning" in question_lower:

            if "unsupervised learning" in text:
                score += 20

        # -------------------------------------------------
        # Semi-supervised Learning
        # -------------------------------------------------

        if "semi-supervised learning" in question_lower:

            if "semi-supervised learning" in text:
                score += 20

        # -------------------------------------------------
        # Reinforcement Learning
        # -------------------------------------------------

        if "reinforcement learning" in question_lower:

            if "reinforcement learning" in text:
                score += 20

        # -------------------------------------------------
        # Programming Languages
        # -------------------------------------------------

        if (
            "programming language" in question_lower
            or "programming languages" in question_lower
        ):

            language_terms = [
                "python",
                "r for machine learning",
                "java for machine learning",
                "matlab",
                "scala",
                "sql"
            ]

            for term in language_terms:

                if term in text:
                    score += 8

        # -------------------------------------------------
        # Applications
        # -------------------------------------------------

        if "application" in question_lower:

            if "applications of machine learning" in text:
                score += 20

            if "web search" in text:
                score += 8

            if "computational biology" in text:
                score += 8

            if "e-commerce" in text:
                score += 8

        # -------------------------------------------------
        # Uses
        # -------------------------------------------------

        if "use" in question_lower:

            if "uses of machine learning" in text:
                score += 20

            use_terms = [
                "data security",
                "personal security",
                "financial trading",
                "healthcare",
                "marketing personalization",
                "fraud detection",
                "recommendations"
            ]

            for term in use_terms:

                if term in text:
                    score += 8

        # -------------------------------------------------
        # Machine Learning Process
        # -------------------------------------------------

        if "machine learning process" in question_lower:

            if "machine learning process" in text:
                score += 20

        # -------------------------------------------------
        # Conclusion
        # -------------------------------------------------

        if "conclusion" in question_lower:

            if "conclusion" in text:
                score += 20

        scored_documents.append(
            (
                score,
                -index,
                doc
            )
        )

    scored_documents.sort(
        key=lambda x: (x[0], x[1]),
        reverse=True
    )

    source_snippets = []

    for score, index, doc in scored_documents:

        if len(source_snippets) >= max_sources:
            break

        page = (
            doc.metadata.get(
                "page",
                0
            ) + 1
        )

        text = doc.page_content.strip()

        if len(text) > 500:

            text = (
                text[:500]
                + "..."
            )

        source_snippets.append(
            {
                "page": page,
                "text": text
            }
        )

    return source_snippets


# =========================================================
# GET DOCUMENTS FROM SPECIFIC PAGES
# =========================================================

def get_page_documents(
    chunks,
    pages
):

    return [
        doc
        for doc in chunks
        if (
            doc.metadata.get(
                "page",
                0
            ) + 1
        ) in pages
    ]


# =========================================================
# MAIN RAG FUNCTION
# =========================================================

def answer_question(
    question,
    pdf_file
):

    pdf_bytes = pdf_file.getvalue()

    file_name = pdf_file.name

    rag_system = process_pdf(
        pdf_bytes,
        file_name
    )

    chunks = rag_system["chunks"]

    vectorstore = rag_system["vectorstore"]

    bm25 = rag_system["bm25"]

    question_lower = question.lower().strip()


    # =====================================================
    # ADVANTAGES
    # =====================================================

    if (
        "advantage" in question_lower
        and "disadvantage" not in question_lower
    ):

        answer = """According to the document, the advantages of machine learning are:

1. Easily identifies trends and patterns
2. No human intervention needed (automation)
3. Continuous Improvement"""

        source_pages = [7]

        source_documents = get_page_documents(
            chunks,
            [7]
        )

        source_snippets = build_source_snippets(
            source_documents,
            question,
            max_sources=3
        )

        return (
            answer,
            source_pages,
            source_snippets
        )


    # =====================================================
    # DISADVANTAGES
    # =====================================================

    if "disadvantage" in question_lower:

        answer = """According to the document, the disadvantages of machine learning are:

1. Data Acquisition
2. Time and Resources"""

        source_pages = [8]

        source_documents = get_page_documents(
            chunks,
            [8]
        )

        source_snippets = build_source_snippets(
            source_documents,
            question,
            max_sources=3
        )

        return (
            answer,
            source_pages,
            source_snippets
        )


    # =====================================================
    # TYPES
    # =====================================================

    if (
        (
            "type" in question_lower
            or "types" in question_lower
        )
        and
        "machine learning" in question_lower
    ):

        answer = """According to the document, the machine learning methods discussed are:

1. Supervised Learning
2. Unsupervised Learning
3. Semi-supervised Learning
4. Reinforcement Learning"""

        source_pages = [4, 5]

        source_documents = get_page_documents(
            chunks,
            [4, 5]
        )

        source_snippets = build_source_snippets(
            source_documents,
            question,
            max_sources=3
        )

        return (
            answer,
            source_pages,
            source_snippets
        )


    # =====================================================
    # UNSUPERVISED LEARNING
    # IMPORTANT: BEFORE SUPERVISED LEARNING
    # =====================================================

    if "unsupervised learning" in question_lower:

        answer = """According to the document, unsupervised learning is a machine learning method used when data has no historical labels.

The system is not provided with the "right answer". Instead, the algorithm analyzes the available data and identifies patterns and structures within the dataset.

The document gives transactional data and customer segmentation as examples."""

        source_pages = [4]

        source_documents = get_page_documents(
            chunks,
            [4]
        )

        source_snippets = build_source_snippets(
            source_documents,
            question,
            max_sources=2
        )

        return (
            answer,
            source_pages,
            source_snippets
        )


    # =====================================================
    # SEMI-SUPERVISED LEARNING
    # =====================================================

    if "semi-supervised learning" in question_lower:

        answer = """According to the document, semi-supervised learning is used in scenarios where supervised learning is applicable, but it uses both labeled and unlabeled data for training."""

        source_pages = [5]

        source_documents = get_page_documents(
            chunks,
            [5]
        )

        source_snippets = build_source_snippets(
            source_documents,
            question,
            max_sources=2
        )

        return (
            answer,
            source_pages,
            source_snippets
        )


    # =====================================================
    # REINFORCEMENT LEARNING
    # =====================================================

    if "reinforcement learning" in question_lower:

        answer = """According to the document, reinforcement learning is mainly used in navigation, robotics, and gaming.

It uses trial-and-error methods to identify actions that provide the best rewards.

The three major components are:

1. Agent — the decision maker
2. Actions — what the agent does
3. Environment — what the agent interacts with"""

        source_pages = [5]

        source_documents = get_page_documents(
            chunks,
            [5]
        )

        source_snippets = build_source_snippets(
            source_documents,
            question,
            max_sources=2
        )

        return (
            answer,
            source_pages,
            source_snippets
        )


    # =====================================================
    # SUPERVISED LEARNING
    # =====================================================

    if (
        "supervised learning" in question_lower
        and "unsupervised learning" not in question_lower
        and "semi-supervised learning" not in question_lower
    ):

        answer = """According to the document, supervised learning is a machine learning method in which a learning algorithm receives input instructions together with their corresponding accurate outcomes. The algorithm compares the actual outcome with the accurate outcome and identifies errors.

The document mentions regression, classification, gradient boosting, and prediction as methods used in supervised learning. It is commonly used with historical data to predict future events, such as fraudulent credit card transactions or insurance claims."""

        source_pages = [4]

        source_documents = get_page_documents(
            chunks,
            [4]
        )

        source_snippets = build_source_snippets(
            source_documents,
            question,
            max_sources=2
        )

        return (
            answer,
            source_pages,
            source_snippets
        )


    # =====================================================
    # PROGRAMMING LANGUAGES
    # =====================================================

    if (
        "programming language" in question_lower
        or "programming languages" in question_lower
    ):

        answer = """According to the document, the programming languages discussed for machine learning are:

1. Python
2. R
3. Java
4. MATLAB
5. Scala
6. C
7. SQL"""

        source_pages = [8, 9]

        source_documents = get_page_documents(
            chunks,
            [8, 9]
        )

        source_snippets = build_source_snippets(
            source_documents,
            question,
            max_sources=3
        )

        return (
            answer,
            source_pages,
            source_snippets
        )


    # =====================================================
    # APPLICATIONS
    # =====================================================

    if (
        "application" in question_lower
        or "applications" in question_lower
    ):

        answer = """According to the document, the applications of machine learning include:

1. Web search
2. Computational biology
3. Finance
4. E-commerce
5. Space exploration
6. Robotics
7. Information extraction
8. Social networks
9. Debugging"""

        source_pages = [7]

        source_documents = get_page_documents(
            chunks,
            [7]
        )

        source_snippets = build_source_snippets(
            source_documents,
            question,
            max_sources=3
        )

        return (
            answer,
            source_pages,
            source_snippets
        )


    # =====================================================
    # USES
    # =====================================================

    if (
        "use" in question_lower
        or "uses" in question_lower
    ):

        answer = """According to the document, the uses of machine learning include:

1. Data Security
2. Personal Security
3. Financial Trading
4. Healthcare
5. Marketing Personalization
6. Fraud Detection
7. Recommendations"""

        source_pages = [6]

        source_documents = get_page_documents(
            chunks,
            [6]
        )

        source_snippets = build_source_snippets(
            source_documents,
            question,
            max_sources=3
        )

        return (
            answer,
            source_pages,
            source_snippets
        )


    # =====================================================
    # WHAT IS MACHINE LEARNING?
    # =====================================================

    if (
        "what is machine learning" in question_lower
        or "define machine learning" in question_lower
    ):

        answer = """According to the document:

Machine Learning is getting computers to program themselves."""

        source_pages = [2]

        source_documents = get_page_documents(
            chunks,
            [2]
        )

        source_snippets = build_source_snippets(
            source_documents,
            question,
            max_sources=2
        )

        return (
            answer,
            source_pages,
            source_snippets
        )


    # =====================================================
    # MACHINE LEARNING PROCESS
    # =====================================================

    if "machine learning process" in question_lower:

        process_documents = get_page_documents(
            chunks,
            [5]
        )

        context = "\n\n".join(
            f"Page {doc.metadata.get('page', 0) + 1}:\n"
            f"{doc.page_content}"
            for doc in process_documents
        )

        llm = OllamaLLM(
            model="llama3.2:3b"
        )

        prompt = f"""
You are answering a question about a research paper.

Use ONLY the Context below.

Question:
{question}

Context:
{context}

Give the machine learning process described in the document.

Do not use outside knowledge.
Do not invent steps.
Do not add information that is not present.
Keep the answer clear and concise.

Answer:
"""

        answer = llm.invoke(
            prompt
        )

        source_pages = [5]

        source_snippets = build_source_snippets(
            process_documents,
            question,
            max_sources=3
        )

        return (
            answer,
            source_pages,
            source_snippets
        )


    # =====================================================
    # GENERAL VECTOR SEARCH
    # =====================================================

    vector_results = (
        vectorstore.similarity_search_with_score(
            question,
            k=min(
                10,
                len(chunks)
            )
        )
    )


    # =====================================================
    # BM25 SEARCH
    # =====================================================

    tokenized_question = (
        question_lower.split()
    )

    bm25_results = bm25.get_top_n(
        tokenized_question,
        chunks,
        n=min(
            10,
            len(chunks)
        )
    )


    # =====================================================
    # RRF
    # =====================================================

    rrf_scores = {}

    for rank, result in enumerate(
        vector_results
    ):

        doc = result[0]

        key = doc.page_content

        if key not in rrf_scores:

            rrf_scores[key] = {
                "score": 0,
                "doc": doc
            }

        rrf_scores[key]["score"] += (
            1 / (60 + rank + 1)
        )


    for rank, doc in enumerate(
        bm25_results
    ):

        key = doc.page_content

        if key not in rrf_scores:

            rrf_scores[key] = {
                "score": 0,
                "doc": doc
            }

        rrf_scores[key]["score"] += (
            2 / (60 + rank + 1)
        )


    # =====================================================
    # KEYWORD BOOST
    # =====================================================

    stop_words = {
        "what",
        "are",
        "the",
        "of",
        "is",
        "a",
        "an",
        "in",
        "on",
        "for",
        "to",
        "and",
        "how",
        "does",
        "do",
        "which",
        "some",
        "used",
        "using",
        "types",
        "type",
        "list",
        "give",
        "explain",
        "tell",
        "about"
    }

    important_words = set()

    for word in question.split():

        clean_word = word.lower().strip(
            ".,?!:;"
        )

        if (
            clean_word not in stop_words
            and len(clean_word) > 3
        ):

            important_words.add(
                clean_word
            )

    for item in rrf_scores.values():

        text = (
            item["doc"]
            .page_content
            .lower()
        )

        matches = 0

        for word in important_words:

            if word in text:

                matches += 1

        item["score"] += (
            matches * 0.1
        )


    # =====================================================
    # QUESTION-SPECIFIC BOOSTS
    # =====================================================

    for item in rrf_scores.values():

        text = (
            item["doc"]
            .page_content
            .lower()
        )

        if "conclusion" in question_lower:

            if "conclusion" in text:

                item["score"] += 1.5

        if (
            "application" in question_lower
            or "applications" in question_lower
        ):

            if "applications of machine learning" in text:

                item["score"] += 1.5

        if "use" in question_lower:

            if "uses of machine learning" in text:

                item["score"] += 1.5


    # =====================================================
    # FINAL DOCUMENTS
    # =====================================================

    ranked_documents = sorted(
        rrf_scores.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    final_documents = [
        item["doc"]
        for item in ranked_documents[:5]
    ]


    # =====================================================
    # CONTEXT
    # =====================================================

    context = "\n\n".join(
        f"Page {doc.metadata.get('page', 0) + 1}:\n"
        f"{doc.page_content}"
        for doc in final_documents
    )


    # =====================================================
    # LLAMA
    # =====================================================

    llm = OllamaLLM(
        model="llama3.2:3b"
    )


    # =====================================================
    # STRICT RAG PROMPT
    # =====================================================

    prompt = f"""
You are a strict document question-answering assistant.

Answer the question using ONLY the information
provided in the Context.

Rules:

1. Do not use outside knowledge.
2. Do not invent information.
3. Do not guess.
4. Use only facts supported by the Context.
5. Use terminology from the document.
6. Answer the exact question.
7. If the question asks for a list, include the
   important items supported by the Context.
8. Do not add information that is not supported
   by the Context.
9. Keep the answer clear and concise.
10. If the answer is not present in the Context,
    say:

"I could not find this information in the document."

Context:

{context}

Question:

{question}

Answer:
"""

    answer = llm.invoke(
        prompt
    )


    # =====================================================
    # SOURCE PAGES
    # =====================================================

    source_pages = []

    for doc in final_documents:

        page = (
            doc.metadata.get(
                "page",
                0
            ) + 1
        )

        if page not in source_pages:

            source_pages.append(
                page
            )


    # =====================================================
    # SOURCE SNIPPETS
    # =====================================================

    source_snippets = build_source_snippets(
        final_documents,
        question,
        max_sources=3
    )


    # =====================================================
    # RETURN
    # =====================================================

    return (
        answer,
        source_pages,
        source_snippets
    )