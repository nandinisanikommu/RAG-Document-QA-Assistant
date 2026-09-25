import streamlit as st
from rag import answer_question, process_pdf_bytes


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="RAG Document Q&A Assistant",
    page_icon="📚",
    layout="centered"
)


# --------------------------------------------------
# Custom Styling
# --------------------------------------------------

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


# --------------------------------------------------
# Cache RAG System
# --------------------------------------------------

@st.cache_resource(
    show_spinner="Indexing document (Vector Embeddings + BM25)..."
)
def get_rag_system(
    file_bytes: bytes,
    file_name: str
) -> dict:

    return process_pdf_bytes(file_bytes)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="main-title">📚 RAG Document Q&A Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Ask dynamic questions about any uploaded PDF document.'
    '</div>',
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Initialize Session State
# --------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "active_file" not in st.session_state:
    st.session_state.active_file = None


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("⚙️ Project Architecture")

    st.markdown(
        """
        - **Embedding:** `all-MiniLM-L6-v2`
        - **Retrieval:** Dense Vector + BM25
        - **Ranking:** RRF + Cross-Encoder
        - **Reranker:** `ms-marco-MiniLM-L-6-v2`
        - **LLM:** Llama 3.2 3B (Ollama)
        - **Framework:** LangChain + Streamlit
        """
    )

    st.divider()

    # Retrieval Settings
    st.subheader("🔍 Retrieval Settings")

    top_k = st.slider(
        "Top-K chunks",
        min_value=3,
        max_value=15,
        value=5,
        step=1,
        help="Number of document chunks retrieved before reranking."
    )

    st.divider()

    # Clear Chat History
    if st.button(
        "🗑️ Clear Chat History",
        use_container_width=True
    ):

        st.session_state.chat_history = []

        st.rerun()


# --------------------------------------------------
# PDF Upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF Document",
    type=["pdf"]
)


# --------------------------------------------------
# Main Application
# --------------------------------------------------

if uploaded_file is not None:

    # --------------------------------------------------
    # Reset Chat When New File Is Uploaded
    # --------------------------------------------------

    if st.session_state.active_file != uploaded_file.name:

        st.session_state.active_file = uploaded_file.name

        st.session_state.chat_history = []


    # --------------------------------------------------
    # Read Uploaded PDF
    # --------------------------------------------------

    file_bytes = uploaded_file.getvalue()


    # --------------------------------------------------
    # Create RAG System
    # --------------------------------------------------

    rag_system = get_rag_system(
        file_bytes,
        uploaded_file.name
    )


    # --------------------------------------------------
    # Active Document
    # --------------------------------------------------

    st.success(
        f"✅ Active Document: **{uploaded_file.name}**"
    )

    st.divider()


    # --------------------------------------------------
    # Display Previous Chat History
    # --------------------------------------------------

    for chat in st.session_state.chat_history:

        # ------------------------------
        # User Message
        # ------------------------------

        with st.chat_message("user"):

            st.write(
                chat["question"]
            )


        # ------------------------------
        # Assistant Message
        # ------------------------------

        with st.chat_message("assistant"):

            st.markdown(
                chat["answer"]
            )


            # ------------------------------
            # Display Sources Only
            # If Sources Exist
            # ------------------------------

            if chat["source_pages"]:

                pages_str = ", ".join(
                    [
                        f"Page {p}"
                        for p in chat["source_pages"]
                    ]
                )

                st.caption(
                    f"📄 **Sources:** {pages_str}"
                )


                # ------------------------------
                # Retrieved Context
                # ------------------------------

                with st.expander(
                    "🔎 View Retrieved Context Snippets"
                ):

                    for snippet in chat["source_snippets"]:

                        st.markdown(
                            f"**Page {snippet['page']}**"
                        )

                        st.caption(
                            snippet["text"]
                        )


# --------------------------------------------------
# Chat Input
# --------------------------------------------------

    if prompt := st.chat_input(
        "Ask a question about this document..."
    ):


        # --------------------------------------------------
        # Display User Message
        # --------------------------------------------------

        with st.chat_message("user"):

            st.write(
                prompt
            )


        # --------------------------------------------------
        # Generate Assistant Response
        # --------------------------------------------------

        with st.chat_message("assistant"):

            with st.spinner(
                "Searching document & generating answer..."
            ):

                try:

                    # ------------------------------
                    # Run RAG Pipeline
                    # ------------------------------

                    answer, source_pages, source_snippets = answer_question(
                        prompt,
                        rag_system,
                        top_k
                    )


                    # ------------------------------
                    # Display Answer
                    # ------------------------------

                    st.markdown(
                        answer
                    )


                    # ------------------------------
                    # Display Sources Only
                    # If Sources Exist
                    # ------------------------------

                    if source_pages:

                        pages_str = ", ".join(
                            [
                                f"Page {p}"
                                for p in source_pages
                            ]
                        )

                        st.caption(
                            f"📄 **Sources:** {pages_str}"
                        )


                        # ------------------------------
                        # Display Retrieved Snippets
                        # ------------------------------

                        with st.expander(
                            "🔎 View Retrieved Context Snippets"
                        ):

                            for snippet in source_snippets:

                                st.markdown(
                                    f"**Page {snippet['page']}**"
                                )

                                st.caption(
                                    snippet["text"]
                                )


                    # ------------------------------
                    # Save Chat History
                    # ------------------------------

                    st.session_state.chat_history.append(
                        {
                            "question": prompt,
                            "answer": answer,
                            "source_pages": source_pages,
                            "source_snippets": source_snippets,
                        }
                    )


                # --------------------------------------------------
                # Error Handling
                # --------------------------------------------------

                except Exception as e:

                    st.error(
                        "❌ An error occurred during retrieval."
                    )

                    st.code(
                        str(e)
                    )


# --------------------------------------------------
# No PDF Uploaded
# --------------------------------------------------

else:

    st.info(
        "👆 Please upload a PDF document above to begin."
    )