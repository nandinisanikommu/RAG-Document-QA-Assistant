from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi


loader = PyPDFLoader("documents/research_paper.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

tokenized_chunks = [
    chunk.page_content.lower().split()
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_chunks)

question = "What types of machine learning are discussed in the conclusion?"

tokenized_question = question.lower().split()

results = bm25.get_top_n(
    tokenized_question,
    chunks,
    n=15
)

print("\n====================================")
print("CONCLUSION QUESTION")
print("====================================")

for rank, doc in enumerate(results):

    page = doc.metadata.get("page", 0) + 1

    print("\nRank:", rank + 1)
    print("Page:", page)

    print(
        doc.page_content[:500].replace("\n", " ")
    )


print("\n====================================")
print("PAGE 12 CONTENT")
print("====================================")

page_12 = documents[11]

print(page_12.page_content)