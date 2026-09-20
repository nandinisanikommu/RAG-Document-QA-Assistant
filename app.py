import streamlit as st
from rag import answer_question, process_pdf_bytes

# Page configuration
st.set_page_config(
    page_title="RAG Document Q&A Assistant", page_icon="📚", layout="centered"
)

# Custom Styling
st.markdown(
    """
<style>
    .main-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .subtitle {
        color: #666;
        font-size: 15px;
        margin-bottom: 20px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Cache vectorstore & BM25 indexing across re-renders
@st.cache_resource(show_spinner="Indexing document (Vector Embeddings + BM25)...")
def get_rag_system(file_bytes: bytes, file_name: str) -> dict:
    return process_pdf_bytes(file_bytes)


# Header
st.markdown(
    '<div class="main-title">📚 RAG Document Q&A Assistant</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Ask dynamic questions about any uploaded PDF document.</div>',
    unsafe_allow_html=True,
)

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "active_file" not in st.session_state:
    st.session_state.active_file = None

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Project Architecture")
    st.markdown(
        """
    - **Embedding:** `all-MiniLM-L6-v2`
    - **Retrieval:** Dense Vector + BM25 (RRF)
    - **LLM:** Llama 3.2 3B (Ollama)
    - **Framework:** LangChain + Streamlit
    """
    )
    st.divider()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Document Upload Section
uploaded_file = st.file_uploader("Upload a PDF Document", type=["pdf"])

if uploaded_file is not None:
    # Reset chat history if a new document is uploaded
    if st.session_state.active_file != uploaded_file.name:
        st.session_state.active_file = uploaded_file.name
        st.session_state.chat_history = []

    # Ingest document and retrieve system components
    file_bytes = uploaded_file.getvalue()
    rag_system = get_rag_system(file_bytes, uploaded_file.name)

    st.success(f"✅ Active Document: **{uploaded_file.name}**")
    st.divider()

    # Render previous conversation history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])

        with st.chat_message("assistant"):
            st.markdown(chat["answer"])
            pages_str = ", ".join([f"Page {p}" for p in chat["source_pages"]])
            st.caption(f"📄 **Sources:** {pages_str}")

            with st.expander("🔎 View Retrieved Context Snippets"):
                for snippet in chat["source_snippets"]:
                    st.markdown(f"**Page {snippet['page']}**")
                    st.caption(snippet["text"])

    # Streamlit Native Chat Input
    if prompt := st.chat_input("Ask a question about this document..."):
        # Render user message
        with st.chat_message("user"):
            st.write(prompt)

        # Generate and render assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching document & generating answer..."):
                try:
                    answer, source_pages, source_snippets = answer_question(
                        prompt, rag_system
                    )

                    st.markdown(answer)
                    pages_str = ", ".join(
                        [f"Page {p}" for p in source_pages]
                    )
                    st.caption(f"📄 **Sources:** {pages_str}")

                    with st.expander("🔎 View Retrieved Context Snippets"):
                        for snippet in source_snippets:
                            st.markdown(f"**Page {snippet['page']}**")
                            st.caption(snippet["text"])

                    # Persist response in session history
                    st.session_state.chat_history.append(
                        {
                            "question": prompt,
                            "answer": answer,
                            "source_pages": source_pages,
                            "source_snippets": source_snippets,
                        }
                    )

                except Exception as e:
                    st.error("❌ An error occurred during retrieval.")
                    st.code(str(e))
else:
    st.info("👆 Please upload a PDF document above to begin.")