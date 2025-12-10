# app.py
"""
Streamlit app that ONLY calls answer_generator.main().

File used:
 - answer_generator.py

No vector DB creation, no manual retrieval, no intermediate calls.
The app:
 - Takes user input
 - Passes it directly to answer_generator.main()
 - Displays the returned answer
"""

import streamlit as st
from pathlib import Path
from answer_generator import main as generate_answer_main


def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []  # {role: 'user'|'assistant', text: str}

    if "vector_db_path" not in st.session_state:
        st.session_state.vector_db_path = "faiss_index"


def render_messages():
    """Display all past messages."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["text"])


def sidebar_controls():
    st.sidebar.header("⚙️ Settings")

    st.session_state.vector_db_path = st.sidebar.text_input(
        "Vector DB Path",
        value=st.session_state.vector_db_path
    )

    # Just warn if DB is missing
    if not Path(st.session_state.vector_db_path).exists():
        st.sidebar.error("❌ Vector DB not found! Run main.py first.")
    else:
        st.sidebar.success("✅ Vector DB is available.")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 About")
    st.sidebar.markdown("""
    This RAG system:
    - Loads PDFs from a folder
    - Chunks and embeds text
    - Stores in FAISS vector DB
    - Retrieves relevant chunks
    - Generates answers using Groq LLM
    """)


def main():
    st.set_page_config(
        page_title="PDF RAG Q&A", 
        page_icon="📄",
        layout="wide"
    )
    
    init_state()
    sidebar_controls()

    st.title("📄 PDF RAG Q&A System")
    st.markdown("Ask questions about your PDF documents — powered by FAISS retrieval and Groq LLM (LLaMA 4 Scout 17B)")

    # Show history
    render_messages()

    # Chat input
    user_question = st.chat_input("Ask your question...")
    if user_question:

        # save user message
        st.session_state.messages.append({"role": "user", "text": user_question})

        # Display user message immediately
        with st.chat_message("user"):
            st.write(user_question)

        # assistant message block
        with st.chat_message("assistant"):

            with st.status("Processing your question..."):
                try:
                    # Call answer_generator.main() directly
                    answer = generate_answer_main(
                        user_question=user_question,
                        vector_db_path=st.session_state.vector_db_path
                    )
                except Exception as e:
                    answer = f"❌ Error: {str(e)}"

            # Display the real answer returned from above
            st.write(answer)
            st.session_state.messages.append(
                {"role": "assistant", "text": answer}
            )


if __name__ == "__main__":
    main()
