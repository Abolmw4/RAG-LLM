from __future__ import annotations
import streamlit as st
from src.ui_service import UIService


st.set_page_config(page_title="RAG Chat", page_icon="🤖", layout="wide",)


@st.cache_resource
def get_ui_service() -> UIService:
    return UIService(config_path="configs/config.yaml")


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_response" not in st.session_state:
        st.session_state.last_response = None


def render_sidebar() -> dict:
    st.sidebar.title("Settings")

    options = {
        "top_k_dense": st.sidebar.slider("Top K Dense", 1, 20, 5),
        "top_k_sparse": st.sidebar.slider("Top K Sparse", 1, 20, 5),
        "fusion_top_k": st.sidebar.slider("Fusion Top K", 1, 20, 5),
        "rerank_top_k": st.sidebar.slider("Rerank Top K", 1, 10, 3),
        "temperature": st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.1),
    }

    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.last_response = None
        st.rerun()
    return options


def render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def render_documents(title: str, documents: list[dict]) -> None:
    if not documents:
        return

    with st.expander(title, expanded=False):
        for doc in documents:
            rank = doc.get("rank", "-")
            score = doc.get("score", "N/A")
            source = doc.get("source", "Unknown")
            text = (
                doc.get("content")
                or doc.get("text")
                or doc.get("page_content")
                or doc.get("document")
                or ""
            )

            st.markdown(f"**Rank:** {rank} | **Score:** {score} | **Source:** {source}")
            st.write(text)
            st.divider()


def render_debug_panels(response: dict) -> None:
    latency = response.get("latency", {})
    if latency:
        with st.expander("Latency", expanded=False):
            st.json(latency)

    render_documents("Retrieved Documents", response.get("retrieved_documents", []))
    render_documents("Reranked Documents", response.get("reranked_documents", []))

    prompt = response.get("prompt")
    if prompt:
        with st.expander("Prompt", expanded=False):
            st.code(prompt, language="text")

    with st.expander("Raw Response", expanded=False):
        st.json(response)


def main() -> None:
    init_session_state()
    options = render_sidebar()
    ui_service = get_ui_service()

    st.title("Hybrid RAG Chat")
    st.caption("Ask questions about your indexed documents.")

    render_chat_history()

    user_query = st.chat_input("Ask something about your documents...")
    if not user_query:
        if st.session_state.last_response:
            render_debug_panels(st.session_state.last_response)
        return

    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ui_service.ask(
                query=user_query,
                chat_history=st.session_state.messages,
                options=options,
            )

        answer = response.get("answer", "No answer returned from pipeline.")
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.last_response = response

    render_debug_panels(response)


if __name__ == "__main__":
    main()
