"""Simple Streamlit chatbot with fixed-topic RAG and web fallback."""

from __future__ import annotations
from pathlib import Path
import streamlit as st
from models.embeddings import load_embedding_model
from models.llm import generate_response, load_model
from utils.rag_utils import (
    create_vector_store,
    load_documents_from_folder,
    retrieve_relevant_context,
    split_documents_into_chunks,
)
from utils.web_search import search_web


DATA_FOLDER = Path("data/documents")


def initialize_session_state() -> None:
    """Create required state keys once."""

    defaults = {
        "messages": [],
        "provider": "OpenAI",
        "response_mode": "Concise",
        "vector_store": None,
        "knowledge_base_ready": False,
        "embedding_ready": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def ensure_knowledge_base() -> tuple[bool, str]:
    """Build and store FAISS index from files in data/documents."""

    DATA_FOLDER.mkdir(parents=True, exist_ok=True)
    documents = load_documents_from_folder(str(DATA_FOLDER))
    if not documents:
        st.session_state.vector_store = None
        st.session_state.knowledge_base_ready = False
        return False, "No context files found in data/documents."

    chunks = split_documents_into_chunks(documents)
    vector_store = create_vector_store(chunks)
    if vector_store is None:
        st.session_state.vector_store = None
        st.session_state.knowledge_base_ready = False
        return False, "Could not build knowledge base from data/documents."

    st.session_state.vector_store = vector_store
    st.session_state.knowledge_base_ready = True
    return True, f"Indexed {len(documents)} file(s) into {len(chunks)} chunk(s)."


def format_retrieved_context(context_items: list[dict]) -> str:
    """Format RAG hits as plain text context for the LLM."""

    if not context_items:
        return ""

    lines: list[str] = []
    for item in context_items:
        lines.append(
            f"Source: {item['source']}\n"
            f"Similarity: {item['similarity']:.3f}\n"
            f"Content: {item['text']}"
        )
    return "\n\n".join(lines)


def build_query_with_memory(query: str, messages: list[dict], window_size: int = 4) -> str:
    """Append recent chat turns so follow-up questions stay coherent."""

    recent = messages[-window_size:] if messages else []
    if not recent:
        return query

    history = "\n".join(f"{turn['role'].capitalize()}: {turn['content']}" for turn in recent)
    return f"Use this history for context:\n{history}\n\nCurrent question: {query}"


def render_sidebar() -> tuple[str, str, int]:
    """Render sidebar controls and return provider, mode, and top_k."""

    with st.sidebar:
        st.header("Configuration")

        provider = st.selectbox("Model Provider", ["OpenAI", "Euron", "Groq", "Gemini"])
        mode = st.radio("Response Mode", ["Concise", "Detailed"], horizontal=True)
        top_k = st.slider("RAG Top K Chunks", min_value=1, max_value=8, value=4)

        st.subheader("Knowledge Base")
        st.caption(
            "Context source: data/documents (txt, md, pdf). "
            "The included file has Machine Learning, Deep Learning, India, IPL, and more."
        )

        if st.button("Rebuild Knowledge Base", use_container_width=True):
            ok, message = ensure_knowledge_base()
            st.success(message) if ok else st.warning(message)

        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        status = "Active (from data folder)" if st.session_state.knowledge_base_ready else "Inactive"
        st.caption(f"RAG status: {status}")

    st.session_state.provider = provider
    st.session_state.response_mode = mode
    return provider, mode, top_k


def render_chat_history() -> None:
    """Render all existing chat messages."""

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def answer_user_query(query: str, provider: str, mode: str, top_k: int) -> str:
    """Generate one assistant response using RAG first, then web search fallback."""

    load_model(provider)

    memory_query = build_query_with_memory(query=query, messages=st.session_state.messages)
    context_items = retrieve_relevant_context(
        query=query,
        vector_store=st.session_state.vector_store,
        top_k=top_k,
    )

    context = format_retrieved_context(context_items) if context_items else search_web(query)
    return generate_response(query=memory_query, context=context, mode=mode)


def main() -> None:
    st.set_page_config(page_title="Simple RAG Chatbot", page_icon="💬", layout="wide")
    st.title("Neostats RAG Chatbot")
    initialize_session_state()

    if not st.session_state.embedding_ready:
        try:
            load_embedding_model()
            st.session_state.embedding_ready = True
        except Exception as exc:
            st.warning(f"Embedding model could not be loaded yet: {exc}")

    if not st.session_state.knowledge_base_ready:
        ok, message = ensure_knowledge_base()
        if not ok:
            st.warning(message)

    provider, mode, top_k = render_sidebar()
    render_chat_history()

    user_query = st.chat_input("Ask a question...")
    if not user_query:
        return

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = answer_user_query(user_query, provider, mode, top_k)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as exc:
                st.error(f"An error occurred while generating the response: {exc}")
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "I ran into an internal error. Please try again.",
                    }
                )


if __name__ == "__main__":
    main()
