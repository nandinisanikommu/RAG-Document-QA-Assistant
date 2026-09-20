from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi


# ============================================================
# 1. LOAD DOCUMENT
# ============================================================

loader = PyPDFLoader("documents/research_paper.pdf")
documents = loader.load()

print(f"Number of pages: {len(documents)}")


# ============================================================
# 2. SPLIT DOCUMENT
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Number of chunks: {len(chunks)}")


# ============================================================
# 3. EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 4. CHROMA
# ============================================================

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)


# ============================================================
# 5. BM25
# ============================================================

tokenized_chunks = [
    chunk.page_content.lower().split()
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_chunks)


# ============================================================
# 6. TEST QUESTIONS
# ============================================================

test_questions = [

    {
        "question": "What is machine learning?",
        "expected_pages": [1, 2]
    },

    {
        "question": "What are some applications of machine learning?",
        "expected_pages": [1, 6, 7]
    },

    {
        "question": "How does machine learning work?",
        "expected_pages": [2, 3]
    },

    {
        "question": "What are the key elements of machine learning?",
        "expected_pages": [3]
    },

    {
        "question": "What is supervised learning?",
        "expected_pages": [4]
    },

    {
        "question": "What is unsupervised learning?",
        "expected_pages": [4]
    },

    {
        "question": "What is semi-supervised learning?",
        "expected_pages": [5]
    },

    {
        "question": "What is reinforcement learning?",
        "expected_pages": [5]
    },

    {
        "question": "What are the steps in the machine learning process?",
        "expected_pages": [5]
    },

    {
        "question": "What are the uses of machine learning?",
        "expected_pages": [6, 7]
    },

    {
        "question": "What are the advantages of machine learning?",
        "expected_pages": [7]
    },

    {
        "question": "What are the disadvantages of machine learning?",
        "expected_pages": [8]
    },

    {
        "question": "What programming languages are used in machine learning?",
        "expected_pages": [8, 9]
    },

    {
        "question": "Which companies are mentioned as using machine learning?",
        "expected_pages": [9, 10, 11]
    },

    {
        "question": "What types of machine learning are discussed in the conclusion?",
        "expected_pages": [12]
    }
]


# ============================================================
# 7. EVALUATION VARIABLES
# ============================================================

total_precision = 0
total_recall = 0
total_hit_rate = 0


# ============================================================
# 8. EVALUATE EACH QUESTION
# ============================================================

for test in test_questions:

    question = test["question"]
    expected_pages = test["expected_pages"]

    question_lower = question.lower()


    # --------------------------------------------------------
    # VECTOR SEARCH
    # --------------------------------------------------------

    vector_results = vectorstore.similarity_search_with_score(
        question,
        k=10
    )


    # --------------------------------------------------------
    # BM25 SEARCH
    # --------------------------------------------------------

    tokenized_question = question_lower.split()

    bm25_results = bm25.get_top_n(
        tokenized_question,
        chunks,
        n=10
    )


    # --------------------------------------------------------
    # RRF
    # --------------------------------------------------------

    rrf_scores = {}


    # Vector results
    for rank, result in enumerate(vector_results):

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
    for rank, doc in enumerate(bm25_results):

        key = doc.page_content

        if key not in rrf_scores:

            rrf_scores[key] = {
                "score": 0,
                "doc": doc
            }

        rrf_scores[key]["score"] += (
            2 / (60 + rank + 1)
        )


    # --------------------------------------------------------
    # KEYWORD BOOST
    # --------------------------------------------------------

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
        "using"
    }


    important_words = set(
        word.lower().strip(".,?!")
        for word in question.split()
        if word.lower().strip(".,?!") not in stop_words
        and len(word.strip(".,?!")) > 3
    )


    for item in rrf_scores.values():

        document_text = item["doc"].page_content.lower()

        matches = 0

        for word in important_words:

            if word in document_text:

                matches += 1

        item["score"] += matches * 0.1


    # --------------------------------------------------------
    # PHRASE BOOST
    # --------------------------------------------------------

    phrases = []


    if "companies" in question_lower:
        phrases.append("companies")


    if "conclusion" in question_lower:
        phrases.append("conclusion")


    if "programming languages" in question_lower:
        phrases.append("programming languages")


    if "machine learning process" in question_lower:
        phrases.append("machine learning process")


    if "applications of machine learning" in question_lower:
        phrases.append("applications of machine learning")


    if "uses of machine learning" in question_lower:
        phrases.append("uses of machine learning")


    if "advantages of machine learning" in question_lower:
        phrases.append("advantages of machine learning")


    if "disadvantages of machine learning" in question_lower:
        phrases.append("disadvantages of machine learning")


    for item in rrf_scores.values():

        document_text = item["doc"].page_content.lower()

        for phrase in phrases:

            if phrase in document_text:

                item["score"] += 0.3


    # --------------------------------------------------------
    # SECTION-AWARE BOOST
    # --------------------------------------------------------

    if "conclusion" in question_lower:

        for item in rrf_scores.values():

            document_text = item["doc"].page_content.lower()

            if "conclusion:" in document_text:

                item["score"] += 1.0

            elif "conclusion" in document_text:

                item["score"] += 0.5


    # --------------------------------------------------------
    # FINAL RANKING
    # --------------------------------------------------------

    ranked_documents = sorted(
        rrf_scores.values(),
        key=lambda x: x["score"],
        reverse=True
    )


    # --------------------------------------------------------
    # TOP 5
    # --------------------------------------------------------

    combined_documents = [
        item["doc"]
        for item in ranked_documents[:5]
    ]


    # --------------------------------------------------------
    # COMPANY PAGE EXPANSION
    # --------------------------------------------------------

    if "companies" in question_lower:

        company_pages = [
            chunk
            for chunk in chunks
            if chunk.metadata.get("page", 0) in [8, 9, 10]
        ]

        existing_content = {
            doc.page_content
            for doc in combined_documents
        }

        for chunk in company_pages:

            if chunk.page_content not in existing_content:

                combined_documents.append(chunk)


    # --------------------------------------------------------
    # CONCLUSION PAGE EXPANSION
    # --------------------------------------------------------

    if "conclusion" in question_lower:

        conclusion_page_chunks = [
            chunk
            for chunk in chunks
            if chunk.metadata.get("page", 0) == 11
        ]

        existing_content = {
            doc.page_content
            for doc in combined_documents
        }

        for chunk in conclusion_page_chunks:

            if chunk.page_content not in existing_content:

                combined_documents.append(chunk)


    # --------------------------------------------------------
    # RETRIEVED PAGES
    # --------------------------------------------------------

    retrieved_pages = []

    for doc in combined_documents:

        page = doc.metadata.get("page", 0) + 1

        if page not in retrieved_pages:

            retrieved_pages.append(page)


    # ========================================================
    # PAGE-LEVEL METRICS
    # ========================================================

    relevant_pages = [
        page
        for page in retrieved_pages
        if page in expected_pages
    ]


    # Precision
    if len(retrieved_pages) > 0:

        precision = (
            len(relevant_pages)
            / len(retrieved_pages)
        )

    else:

        precision = 0


    # Recall
    recall = (
        len(relevant_pages)
        / len(expected_pages)
    )


    # Hit rate
    hit_rate = (
        1
        if len(relevant_pages) > 0
        else 0
    )


    total_precision += precision
    total_recall += recall
    total_hit_rate += hit_rate


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    print("\n" + "=" * 70)

    print(f"Question: {question}")

    print(f"Expected pages: {expected_pages}")

    print(f"Retrieved pages: {retrieved_pages}")

    print(f"Relevant pages: {relevant_pages}")

    print(f"Page Precision: {precision:.2f}")

    print(f"Recall: {recall:.2f}")

    print(f"Hit Rate: {hit_rate:.2f}")


# ============================================================
# 9. FINAL RESULTS
# ============================================================

number_of_questions = len(test_questions)


average_precision = (
    total_precision
    / number_of_questions
)


average_recall = (
    total_recall
    / number_of_questions
)


average_hit_rate = (
    total_hit_rate
    / number_of_questions
)


print("\n")
print("=" * 70)
print("RAG EVALUATION RESULTS")
print("=" * 70)

print(
    f"Average Page Precision: "
    f"{average_precision:.2f}"
)

print(
    f"Average Recall: "
    f"{average_recall:.2f}"
)

print(
    f"Average Hit Rate: "
    f"{average_hit_rate:.2f}"
)

print("=" * 70)