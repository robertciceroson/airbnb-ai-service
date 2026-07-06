"""
RAG ingest pipeline: loads policy documents → chunks → embeds → saves FAISS index.

Usage:
    python ingest_policies.py

Place .txt or .md policy files in data/policies/ before running.
The script also fetches key Airbnb Help Center URLs if requests is available.
"""
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path
import os

from app.config import settings


def build_vector_store(docs_dir: Path | None = None, index_path: Path | None = None):
    docs_dir   = docs_dir   or settings.policies_dir
    index_path = index_path or settings.faiss_index_path

    # ── Load documents ────────────────────────────────────────────────────────
    loader = DirectoryLoader(
        str(docs_dir),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()

    if not documents:
        raise ValueError(
            f"No .txt files found in {docs_dir}. "
            "Run ingest_policies.py to fetch Airbnb Help Center content first."
        )

    print(f"📄 Loaded {len(documents)} policy document(s).")

    # ── Chunk ─────────────────────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    print(f"✂️  Split into {len(chunks)} chunks.")

    # ── Embed + index ─────────────────────────────────────────────────────────
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    vector_store = FAISS.from_documents(chunks, embeddings)

    index_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(index_path))
    print(f"✅ FAISS index saved → {index_path}")

    return vector_store


def load_vector_store(index_path: Path | None = None) -> FAISS:
    index_path = index_path or settings.faiss_index_path

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. "
            "Run `python ingest_policies.py` first."
        )

    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    vs = FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)
    print("✅ FAISS vector store loaded.")
    return vs


def get_retriever(vector_store: FAISS):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.retriever_k},
    )
