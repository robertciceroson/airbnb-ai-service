"""
RAG retrieval using BM25 (sparse keyword retrieval) — no PyTorch/embeddings needed.
Documents are loaded from data/policies/ and indexed in memory at startup.
This keeps the memory footprint well within Render's free tier (512MB).
"""
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

from app.config import settings


def _load_chunks(docs_dir: Path | None = None) -> list:
    docs_dir = docs_dir or settings.policies_dir

    if not docs_dir.exists() or not any(docs_dir.glob("**/*.txt")):
        raise FileNotFoundError(
            f"No policy .txt files found in {docs_dir}. "
            "Run `python ingest_policies.py` first."
        )

    loader = DirectoryLoader(
        str(docs_dir),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    print(f"📄 Loaded {len(documents)} policy document(s).")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    print(f"✂️  Split into {len(chunks)} chunks.")
    return chunks


def get_retriever(docs_dir: Path | None = None) -> BM25Retriever:
    """Build and return a BM25 retriever from policy documents."""
    chunks = _load_chunks(docs_dir)
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = settings.retriever_k
    print("✅ BM25 retriever ready.")
    return retriever
