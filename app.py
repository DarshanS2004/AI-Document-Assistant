from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from assistant_core import (
    OpenRouterEmbeddings,
    answer_question,
    build_vector_store,
    compute_file_digest,
    load_documents,
    load_vector_store,
    persist_uploaded_files,
    save_index_manifest,
    search_web,
    split_documents,
)

APP_TITLE = "AI Document Assistant"
APP_URL = os.getenv("OPENROUTER_APP_URL", "http://localhost:8501")
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "OPENROUTER_EMBEDDING_MODEL",
    "openai/text-embedding-3-small",
)

WORKSPACE_DIR = Path(__file__).resolve().parent
CACHE_DIR = WORKSPACE_DIR / ".cache"
UPLOADS_DIR = CACHE_DIR / "uploads"
INDICES_DIR = CACHE_DIR / "indices"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("vector_store", None)
    st.session_state.setdefault("doc_source", None)
    st.session_state.setdefault("doc_signature", None)
    st.session_state.setdefault("last_retrieved_docs", [])
    st.session_state.setdefault("last_web_results", [])


def get_secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)


def reset_knowledge_base() -> None:
    st.session_state["vector_store"] = None
    st.session_state["doc_source"] = None
    st.session_state["doc_signature"] = None
    st.session_state["last_retrieved_docs"] = []
    st.session_state["last_web_results"] = []


def build_or_load_knowledge_base(
    uploaded_file,
    api_key: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    if not uploaded_file:
        raise ValueError("Upload one document to build the knowledge base.")

    doc_signature = compute_file_digest(uploaded_file.name, uploaded_file.getvalue())

    if st.session_state.get("doc_signature") == doc_signature and st.session_state.get(
        "vector_store"
    ) is not None:
        return

    embeddings = OpenRouterEmbeddings(api_key=api_key, model=embedding_model)
    upload_dir = UPLOADS_DIR / doc_signature
    index_dir = INDICES_DIR / doc_signature

    if (index_dir / "index.faiss").exists() and (index_dir / "index.pkl").exists():
        vector_store = load_vector_store(index_dir, embeddings)
    else:
        saved_paths = persist_uploaded_files([uploaded_file], upload_dir)
        documents = load_documents(saved_paths)
        chunks = split_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        vector_store = build_vector_store(chunks, embeddings, index_dir)
        save_index_manifest(
            index_dir,
            {
                "file": saved_paths[0].name,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "chunks_indexed": len(chunks),
            },
        )

    st.session_state["vector_store"] = vector_store
    st.session_state["doc_source"] = uploaded_file.name
    st.session_state["doc_signature"] = doc_signature


def render_sidebar():
    st.sidebar.title("Control Panel")
    st.sidebar.caption("Upload one document, choose models, and turn on web enrichment.")

    api_key = st.sidebar.text_input(
        "OpenRouter API key",
        value=get_secret("OPENROUTER_API_KEY"),
        type="password",
        help="Used for both answer generation and embeddings.",
    )
    model_name = st.sidebar.text_input("Chat model", value=DEFAULT_MODEL)
    embedding_model = st.sidebar.text_input(
        "Embedding model",
        value=DEFAULT_EMBEDDING_MODEL,
    )
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    chunk_size = st.sidebar.slider("Chunk size", 500, 2000, 1200, 100)
    chunk_overlap = st.sidebar.slider("Chunk overlap", 0, 400, 180, 20)
    doc_k = st.sidebar.slider("Document chunks per answer", 2, 8, 4, 1)
    web_k = st.sidebar.slider("Web results", 0, 8, 4, 1)
    use_web = st.sidebar.checkbox("Enrich answers with live web search", value=True)

    uploaded_file = st.sidebar.file_uploader(
        "Upload one document",
        type=["pdf", "docx", "txt", "md"],
    )

    col1, col2 = st.sidebar.columns(2)
    build_clicked = col1.button("Build index", use_container_width=True)
    clear_clicked = col2.button("Clear chat", use_container_width=True)

    if clear_clicked:
        st.session_state["messages"] = []

    if st.sidebar.button("Reset knowledge base", use_container_width=True):
        reset_knowledge_base()

    with st.sidebar.expander("How this app answers", expanded=False):
        st.markdown(
            """
            - It indexes your uploaded document into a vector store.
            - For each question, it retrieves the most relevant chunks first.
            - If web enrichment is enabled, it adds live search snippets.
            - The model then writes one answer that stays grounded in your document.
            """
        )

    return {
        "api_key": api_key,
        "model_name": model_name,
        "embedding_model": embedding_model,
        "temperature": temperature,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "doc_k": doc_k,
        "web_k": web_k,
        "use_web": use_web,
        "uploaded_file": uploaded_file,
        "build_clicked": build_clicked,
    }


def render_header() -> None:
    st.title(APP_TITLE)
    st.caption(
        "Ask questions against your document, then optionally blend in live web context without losing document grounding."
    )

    if st.session_state.get("doc_source"):
        st.success("Knowledge base ready for: " + st.session_state["doc_source"])
    else:
        st.info("Upload one document and click Build index to get started.")


def render_chat() -> None:
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def render_sources() -> None:
    docs = st.session_state.get("last_retrieved_docs", [])
    web_results = st.session_state.get("last_web_results", [])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Retrieved document chunks")
        if not docs:
            st.caption("No document chunks retrieved yet.")
        for idx, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source_name", "Uploaded document")
            page = doc.metadata.get("page")
            label = f"{source} | page {page + 1}" if isinstance(page, int) else source
            with st.expander(f"Doc {idx}: {label}", expanded=False):
                st.write(doc.page_content)

    with col2:
        st.subheader("Web context")
        if not web_results:
            st.caption("No live web results used for the latest answer.")
        for idx, result in enumerate(web_results, start=1):
            with st.expander(f"Web {idx}: {result['title']}", expanded=False):
                st.markdown(f"[Open source]({result['href']})")
                st.write(result["body"])


def main() -> None:
    init_state()
    config = render_sidebar()
    render_header()
    render_chat()

    if config["build_clicked"]:
        if not config["api_key"]:
            st.error("Add your OpenRouter API key first.")
        else:
            with st.spinner("Building the document index..."):
                try:
                    build_or_load_knowledge_base(
                        uploaded_file=config["uploaded_file"],
                        api_key=config["api_key"],
                        embedding_model=config["embedding_model"],
                        chunk_size=config["chunk_size"],
                        chunk_overlap=config["chunk_overlap"],
                    )
                except Exception as exc:
                    st.error(f"Could not build the knowledge base: {exc}")
                else:
                    st.success("Your document index is ready.")

    question = st.chat_input("Ask something about your document...")
    if not question:
        render_sources()
        return

    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    if st.session_state.get("vector_store") is None:
        warning = "Build the knowledge base before asking questions."
        st.session_state["messages"].append({"role": "assistant", "content": warning})
        with st.chat_message("assistant"):
            st.warning(warning)
        render_sources()
        return

    with st.chat_message("assistant"):
        with st.spinner("Searching your document and drafting the answer..."):
            try:
                retrieved_docs = st.session_state["vector_store"].similarity_search(
                    question,
                    k=config["doc_k"],
                )
                web_results = (
                    search_web(question, max_results=config["web_k"])
                    if config["use_web"] and config["web_k"] > 0
                    else []
                )

                answer = answer_question(
                    api_key=config["api_key"],
                    model_name=config["model_name"],
                    temperature=config["temperature"],
                    app_title=APP_TITLE,
                    app_url=APP_URL,
                    question=question,
                    retrieved_docs=retrieved_docs,
                    web_results=web_results,
                    chat_history=st.session_state["messages"][:-1],
                )
            except Exception as exc:
                answer = (
                    "I hit an error while preparing the answer.\n\n"
                    f"`{exc}`"
                )
                retrieved_docs = []
                web_results = []

        st.markdown(answer)

    st.session_state["last_retrieved_docs"] = retrieved_docs
    st.session_state["last_web_results"] = web_results
    st.session_state["messages"].append({"role": "assistant", "content": answer})

    render_sources()


if __name__ == "__main__":
    main()
