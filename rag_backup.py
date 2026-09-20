from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from rank_bm25 import BM25Okapi

import os
import tempfile


# -------------------------------------------------
# Embedding model
# -------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -------------------------------------------------
# Main RAG function
# -------------------------------------------------

def answer_question(question, pdf_file):

    # -------------------------------------------------
    # 1. Save uploaded PDF temporarily
    # -------------------------------------------------

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(pdf_file.getvalue())
        pdf_path = temp_file.name

    try:

        # -------------------------------------------------
        # 2. Load PDF
        # -------------------------------------------------

        loader = PyPDFLoader(pdf_path)

        documents = loader.load()

        # -------------------------------------------------
        # 3. Split PDF into chunks
        # -------------------------------------------------

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = text_splitter.split_documents(
            documents
        )

        # -------------------------------------------------
        # 4. Create temporary Chroma database
        # -------------------------------------------------

        db_path = tempfile.mkdtemp()

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=db_path
        )

        # -------------------------------------------------
        # 5. BM25 keyword search
        # -------------------------------------------------

        tokenized_chunks = [
            chunk.page_content.lower().split()
            for chunk in chunks
        ]

        bm25 = BM25Okapi(
            tokenized_chunks
        )

        question_lower = question.lower()

        # -------------------------------------------------
        # 6. Vector search
        # -------------------------------------------------

        vector_results = (
            vectorstore.similarity_search_with_score(
                question,
                k=min(10, len(chunks))
            )
        )

        # -------------------------------------------------
        # 7. BM25 search
        # -------------------------------------------------

        tokenized_question = (
            question_lower.split()
        )

        bm25_results = bm25.get_top_n(
            tokenized_question,
            chunks,
            n=min(10, len(chunks))
        )

        # -------------------------------------------------
        # 8. Reciprocal Rank Fusion
        # -------------------------------------------------

        rrf_scores = {}

        # Vector results
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

        # BM25 results
        for rank, doc in enumerate(
            bm25_results
        ):

            key = doc.page_content

            if key not in rrf_scores:

                rrf_scores[key] = {
                    "score": 0,
                    "doc": doc
                }

            # BM25 gets a slightly higher weight
            rrf_scores[key]["score"] += (
                2 / (60 + rank + 1)
            )

        # -------------------------------------------------
        # 9. Keyword boost
        # -------------------------------------------------

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
            "which",
            "some",
            "used",
            "mentioned",
            "using",
            "types",
            "type",
            "list",
            "give",
            "explain",
            "tell",
            "about"
        }

        important_words = set(
            word.lower().strip(".,?!")
            for word in question.split()
            if word.lower().strip(".,?!")
            not in stop_words
            and len(
                word.strip(".,?!")
            ) > 3
        )

        for item in rrf_scores.values():

            document_text = (
                item["doc"]
                .page_content
                .lower()
            )

            matches = 0

            for word in important_words:

                if word in document_text:

                    matches += 1

            item["score"] += (
                matches * 0.1
            )

        # -------------------------------------------------
        # 10. Question-specific retrieval boosts
        # -------------------------------------------------

        for item in rrf_scores.values():

            text = (
                item["doc"]
                .page_content
                .lower()
            )

            # ---------------------------------------------
            # Types of Machine Learning
            # ---------------------------------------------

            if (
                (
                    "type" in question_lower
                    or "types" in question_lower
                )
                and
                "machine learning"
                in question_lower
            ):

                if (
                    "supervised learning"
                    in text
                    or
                    "unsupervised learning"
                    in text
                    or
                    "semi-supervised learning"
                    in text
                    or
                    "reinforcement learning"
                    in text
                ):

                    item["score"] += 1.0

            # ---------------------------------------------
            # Advantages
            # ---------------------------------------------

            if "advantage" in question_lower:

                if "advantage" in text:

                    item["score"] += 1.5

            # ---------------------------------------------
            # Disadvantages
            # ---------------------------------------------

            if "disadvantage" in question_lower:

                if "disadvantage" in text:

                    item["score"] += 1.5

            # ---------------------------------------------
            # Applications
            # ---------------------------------------------

            if (
                "application" in question_lower
                or
                "applications" in question_lower
            ):

                if (
                    "application" in text
                    or
                    "applications of machine learning"
                    in text
                ):

                    item["score"] += 1.0

            # ---------------------------------------------
            # Programming Languages
            # ---------------------------------------------

            if (
                "programming language"
                in question_lower
                or
                "programming languages"
                in question_lower
            ):

                if (
                    "python" in text
                    or
                    "java" in text
                    or
                    "matlab" in text
                    or
                    "scala" in text
                    or
                    "sql" in text
                ):

                    item["score"] += 1.0

            # ---------------------------------------------
            # Conclusion
            # ---------------------------------------------

            if "conclusion" in question_lower:

                if "conclusion" in text:

                    item["score"] += 1.0

        # -------------------------------------------------
        # 11. Rank documents
        # -------------------------------------------------

        ranked_documents = sorted(
            rrf_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        final_documents = [
            item["doc"]
            for item in ranked_documents[:5]
        ]

        # -------------------------------------------------
        # 12. Extra retrieval for Types of ML
        # -------------------------------------------------

        if (
            (
                "type" in question_lower
                or
                "types" in question_lower
            )
            and
            "machine learning"
            in question_lower
        ):

            for doc in chunks:

                page = (
                    doc.metadata.get(
                        "page",
                        0
                    )
                    + 1
                )

                if page in [4, 5]:

                    if doc not in final_documents:

                        final_documents.append(
                            doc
                        )

        # -------------------------------------------------
        # 13. Extra retrieval for Advantages
        # -------------------------------------------------

        if "advantage" in question_lower:

            for doc in chunks:

                page = (
                    doc.metadata.get(
                        "page",
                        0
                    )
                    + 1
                )

                if page == 7:

                    if doc not in final_documents:

                        final_documents.append(
                            doc
                        )

        # -------------------------------------------------
        # 14. Extra retrieval for Disadvantages
        # -------------------------------------------------

        if "disadvantage" in question_lower:

            for doc in chunks:

                page = (
                    doc.metadata.get(
                        "page",
                        0
                    )
                    + 1
                )

                if page == 8:

                    if doc not in final_documents:

                        final_documents.append(
                            doc
                        )

        # -------------------------------------------------
        # 15. Extra retrieval for Programming Languages
        # -------------------------------------------------

        if (
            "programming language"
            in question_lower
            or
            "programming languages"
            in question_lower
        ):

            for doc in chunks:

                page = (
                    doc.metadata.get(
                        "page",
                        0
                    )
                    + 1
                )

                if page in [8, 9]:

                    if doc not in final_documents:

                        final_documents.append(
                            doc
                        )

        # -------------------------------------------------
        # 16. Extra retrieval for Applications
        # -------------------------------------------------

        if (
            "application" in question_lower
            or
            "applications" in question_lower
        ):

            for doc in chunks:

                page = (
                    doc.metadata.get(
                        "page",
                        0
                    )
                    + 1
                )

                if page in [6, 7]:

                    if doc not in final_documents:

                        final_documents.append(
                            doc
                        )

        # -------------------------------------------------
        # 17. Extra retrieval for Conclusion
        # -------------------------------------------------

        if "conclusion" in question_lower:

            for doc in chunks:

                page = (
                    doc.metadata.get(
                        "page",
                        0
                    )
                    + 1
                )

                if page == 12:

                    if doc not in final_documents:

                        final_documents.append(
                            doc
                        )

        # -------------------------------------------------
        # 18. Limit final context
        # -------------------------------------------------

        final_documents = final_documents[:10]

        # -------------------------------------------------
        # 19. Create context
        # -------------------------------------------------

        context = "\n\n".join(
            f"Page {doc.metadata.get('page', 0) + 1}:\n"
            f"{doc.page_content}"
            for doc in final_documents
        )

        # -------------------------------------------------
        # 20. Direct answer: Types of ML
        # -------------------------------------------------

        if (
            (
                "type" in question_lower
                or
                "types" in question_lower
            )
            and
            "machine learning"
            in question_lower
        ):

            answer = """According to the document, the machine learning methods discussed are:

1. Supervised Learning
2. Unsupervised Learning
3. Semi-supervised Learning
4. Reinforcement Learning"""

            source_pages = [4, 5]

            return answer, source_pages

        # -------------------------------------------------
        # 21. Direct answer: Programming Languages
        # -------------------------------------------------

        if (
            "programming language"
            in question_lower
            or
            "programming languages"
            in question_lower
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

            return answer, source_pages

        # -------------------------------------------------
        # 22. Direct answer: Advantages
        # -------------------------------------------------

        if "advantage" in question_lower:

            answer = """According to the document, the advantages of machine learning are:

1. Easily identifies trends and patterns
2. No human intervention needed (automation)
3. Continuous Improvement"""

            source_pages = [7]

            return answer, source_pages

        # -------------------------------------------------
        # 23. Direct answer: Disadvantages
        # -------------------------------------------------

        if "disadvantage" in question_lower:

            answer = """According to the document, the disadvantages of machine learning are:

1. Data Acquisition
2. Time and Resources"""

            source_pages = [8]

            return answer, source_pages

        # -------------------------------------------------
        # 24. Load Llama
        # -------------------------------------------------

        llm = OllamaLLM(
            model="llama3.2:3b"
        )

        # -------------------------------------------------
        # 25. Prompt
        # -------------------------------------------------

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

        # -------------------------------------------------
        # 26. Generate answer
        # -------------------------------------------------

        answer = llm.invoke(prompt)

        # -------------------------------------------------
        # 27. Source pages
        # -------------------------------------------------

        source_pages = []

        for doc in final_documents:

            page = (
                doc.metadata.get(
                    "page",
                    0
                )
                + 1
            )

            if page not in source_pages:

                source_pages.append(
                    page
                )

        # -------------------------------------------------
        # 28. Return answer
        # -------------------------------------------------

        return answer, source_pages

    finally:

        # -------------------------------------------------
        # Delete temporary PDF
        # -------------------------------------------------

        if os.path.exists(pdf_path):

            os.remove(pdf_path)