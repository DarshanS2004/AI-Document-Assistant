from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from ddgs import DDGS
from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openrouter import ChatOpenRouter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

load_dotenv()


class OpenRouterEmbeddings(Embeddings):
    """Embeddings wrapper that uses OpenRouter's OpenAI-compatible embeddings API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        app_title: str = "AI Document Assistant",
        app_url: str = "http://localhost:8501",
        batch_size: int = 32,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers={
                "HTTP-Referer": app_url,
                "X-Title": app_title,
            },
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        cleaned = [text.replace("\n", " ").strip() for text in texts]
        cleaned = [text for text in cleaned if text]
        if not cleaned:
            return []

        embeddings: list[list[float]] = []
        for start in range(0, len(cleaned), self.batch_size):
            batch = cleaned[start : start + self.batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)
            embeddings.extend(item.embedding for item in response.data)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=text.replace("\n", " ").strip(),
        )
        return response.data[0].embedding


def compute_file_digest(file_name: str, payload: bytes) -> str:
    hasher = hashlib.sha256()
    hasher.update(file_name.encode("utf-8"))
    hasher.update(payload)
    return hasher.hexdigest()


def persist_uploaded_files(files: Iterable, target_dir: Path) -> list[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for uploaded_file in files:
        payload = uploaded_file.getbuffer()
        path = target_dir / uploaded_file.name
        path.write_bytes(payload)
        paths.append(path)
    return paths


def load_documents(file_paths: Iterable[Path]) -> list[Document]:
    documents: list[Document] = []

    for file_path in file_paths:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif suffix == ".docx":
            loader = Docx2txtLoader(str(file_path))
        elif suffix in {".txt", ".md"}:
            loader = TextLoader(str(file_path), encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {file_path.name}")

        loaded_docs = loader.load()
        for doc in loaded_docs:
            doc.metadata["source_name"] = file_path.name
            doc.metadata["source_path"] = str(file_path)
        documents.extend(loaded_docs)

    return documents


def split_documents(
    documents: list[Document],
    *,
    chunk_size: int = 1200,
    chunk_overlap: int = 180,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def build_vector_store(
    documents: list[Document],
    embeddings: Embeddings,
    index_dir: Path,
) -> FAISS:
    index_dir.parent.mkdir(parents=True, exist_ok=True)
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(str(index_dir))
    return vector_store


def load_vector_store(index_dir: Path, embeddings: Embeddings) -> FAISS:
    return FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def create_llm(
    *,
    api_key: str,
    model: str,
    temperature: float,
    app_title: str,
    app_url: str,
) -> ChatOpenRouter:
    return ChatOpenRouter(
        model=model,
        api_key=api_key,
        temperature=temperature,
        app_title=app_title,
        app_url=app_url,
    )


def search_web(query: str, max_results: int = 5) -> list[dict[str, str]]:
    results = DDGS(timeout=12).text(query, max_results=max_results)
    normalized: list[dict[str, str]] = []

    for item in results:
        normalized.append(
            {
                "title": item.get("title", "Untitled result"),
                "href": item.get("href") or item.get("url", ""),
                "body": item.get("body", ""),
            }
        )

    return normalized


def format_documents_for_prompt(documents: list[Document]) -> str:
    if not documents:
        return "No matching document context was retrieved."

    lines: list[str] = []
    for idx, doc in enumerate(documents, start=1):
        source_name = doc.metadata.get("source_name", "Uploaded document")
        page = doc.metadata.get("page")
        page_label = f", page {page + 1}" if isinstance(page, int) else ""
        content = " ".join(doc.page_content.split())
        lines.append(f"[Doc {idx}] {source_name}{page_label}\n{content}")
    return "\n\n".join(lines)


def format_web_results_for_prompt(results: list[dict[str, str]]) -> str:
    if not results:
        return "No web context was retrieved."

    lines: list[str] = []
    for idx, item in enumerate(results, start=1):
        lines.append(
            "\n".join(
                [
                    f"[Web {idx}] {item['title']}",
                    f"URL: {item['href']}",
                    f"Snippet: {item['body']}",
                ]
            )
        )
    return "\n\n".join(lines)


def format_chat_history(messages: list[dict[str, str]], limit: int = 6) -> str:
    if not messages:
        return "No prior conversation."

    window = messages[-limit:]
    return "\n".join(
        f"{message['role'].capitalize()}: {message['content']}" for message in window
    )


def answer_question(
    *,
    api_key: str,
    model_name: str,
    temperature: float,
    app_title: str,
    app_url: str,
    question: str,
    retrieved_docs: list[Document],
    web_results: list[dict[str, str]],
    chat_history: list[dict[str, str]],
) -> str:
    llm = create_llm(
        api_key=api_key,
        model=model_name,
        temperature=temperature,
        app_title=app_title,
        app_url=app_url,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an AI document assistant. "
                    "Answer the user with the uploaded document as the primary source of truth. "
                    "If web results are provided, use them only to enrich or compare against the document. "
                    "Be transparent when the document does not contain the answer. "
                    "Do not invent facts or fake citations. "
                    "Respond in markdown and use short sections when helpful."
                ),
            ),
            (
                "human",
                (
                    "Conversation history:\n{chat_history}\n\n"
                    "User question:\n{question}\n\n"
                    "Document context:\n{document_context}\n\n"
                    "Web context:\n{web_context}\n\n"
                    "Write the answer with this structure:\n"
                    "1. A direct answer.\n"
                    "2. A short 'From your document' section.\n"
                    "3. If helpful, a short 'Additional web context' section.\n"
                    "4. A brief 'Sources used' list citing Doc/Web labels."
                ),
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    return chain.invoke(
        {
            "chat_history": format_chat_history(chat_history),
            "question": question,
            "document_context": format_documents_for_prompt(retrieved_docs),
            "web_context": format_web_results_for_prompt(web_results),
        }
    )


def save_index_manifest(index_dir: Path, metadata: dict) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = index_dir / "manifest.json"
    manifest_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
