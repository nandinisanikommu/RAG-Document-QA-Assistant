import streamlit as st
from rag import answer_question

# Page configuration
st.set_page_config(
    page_title="RAG Document Q&A Assistant",
    page_icon="📚",
    layout="centered"
)

# Custom styling
st.markdown("""
<style>
    .main-title {
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #666;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .source-box {
        padding: 12px;
        border-radius: 8px;
        background-color: #f5f5f5;
        margin-bottom: 8px;
    }

    .answer-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #f8f9fa;
        border-left: 4px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)


# Title
st.markdown(
    '<div class="main-title">📚 RAG Document Q&A Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions about your PDF using Retrieval-Augmented Generation.'
    '</div>',
    unsafe_allow_html=True
)


# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Sidebar
with st.sidebar:
    st.header("⚙️ Project Information")

    st.write("**Technology:** RAG")
    st.write("**Embedding Model:**")
    st.caption("all-MiniLM-L6-v2")

    st.write("**Retrieval:**")
    st.caption("Vector Search + BM25 + RRF")

    st.write("**LLM:**")
    st.caption("Llama 3.2 3B")

    st.write("**Interface:**")
    st.caption("Streamlit")

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# Upload section
st.subheader("📄 Upload Document")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)

if uploaded_file is not None:

    st.success(f"✅ Document loaded: {uploaded_file.name}")

    st.divider()

    # Question section
    st.subheader("💬 Ask a Question")

    question = st.text_input(
        "Enter your question",
        placeholder="Example: What are the advantages of machine learning?"
    )

    if st.button(
        "🔍 Ask Question",
        use_container_width=True
    ):

        if not question.strip():

            st.warning("⚠️ Please enter a question.")

        else:

            with st.spinner(
                "🔎 Searching document and generating answer..."
            ):

                try:

                    answer, source_pages, source_snippets = answer_question(
                        question,
                        uploaded_file
                    )

                    # Save conversation
                    st.session_state.chat_history.append(
                        {
                            "question": question,
                            "answer": answer,
                            "source_pages": source_pages,
                            "source_snippets": source_snippets
                        }
                    )

                except Exception as e:

                    st.error("❌ Something went wrong.")
                    st.code(str(e))


# Conversation history
if st.session_state.chat_history:

    st.divider()

    st.subheader("💬 Conversation")

    for chat in st.session_state.chat_history:

        # Question
        st.markdown(
            f"**🧑 You:** {chat['question']}"
        )

        # Answer
        st.markdown("**🤖 Assistant:**")

        st.markdown(
            f'<div class="answer-box">{chat["answer"]}</div>',
            unsafe_allow_html=True
        )

        # Source pages
        st.markdown("**📄 Source Pages:**")

        pages = ", ".join(
            [f"Page {page}" for page in chat["source_pages"]]
        )

        st.write(pages)

        # Retrieved sources
        with st.expander("🔎 View Retrieved Sources"):

            for source in chat["source_snippets"][:3]:

                st.markdown(
                    f"**Page {source['page']}**"
                )

                st.caption(
                    source["text"]
                )

        st.divider()