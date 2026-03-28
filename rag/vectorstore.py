import json
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from pypdf import PdfReader

from config import (
    CHROMA_PERSIST_DIR,
    KNOWLEDGE_BASE_CHUNK_OVERLAP,
    KNOWLEDGE_BASE_CHUNK_SIZE,
    KNOWLEDGE_BASE_DIR,
)
from rag.embeddings import get_embeddings


SUPPORTED_KNOWLEDGE_FILE_TYPES = (".txt", ".pdf")
INGESTION_MANIFEST = Path(CHROMA_PERSIST_DIR) / "ingestion_manifest.json"


def _knowledge_base_manifest() -> dict:
    kb_root = Path(KNOWLEDGE_BASE_DIR)
    files = []
    for path in sorted(kb_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_KNOWLEDGE_FILE_TYPES:
            stat = path.stat()
            files.append(
                {
                    "path": path.relative_to(kb_root).as_posix(),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )

    return {
        "knowledge_base_dir": kb_root.resolve().as_posix(),
        "chunk_size": KNOWLEDGE_BASE_CHUNK_SIZE,
        "chunk_overlap": KNOWLEDGE_BASE_CHUNK_OVERLAP,
        "files": files,
    }


def _manifest_is_current() -> bool:
    if not INGESTION_MANIFEST.exists():
        return False

    try:
        saved_manifest = json.loads(INGESTION_MANIFEST.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False

    return saved_manifest == _knowledge_base_manifest()


def _load_knowledge_documents():
    documents = []
    kb_root = Path(KNOWLEDGE_BASE_DIR)

    for path in sorted(kb_root.rglob("*")):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        if suffix == ".txt":
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": path.relative_to(kb_root).as_posix()},
                    )
                )
        elif suffix == ".pdf":
            reader = PdfReader(str(path))
            source = path.relative_to(kb_root).as_posix()
            for page_number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={"source": source, "page": page_number},
                        )
                    )

    return [doc for doc in documents if doc.page_content.strip()]


def _chunk_text(text: str) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []

    if KNOWLEDGE_BASE_CHUNK_OVERLAP >= KNOWLEDGE_BASE_CHUNK_SIZE:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    step = KNOWLEDGE_BASE_CHUNK_SIZE - KNOWLEDGE_BASE_CHUNK_OVERLAP
    chunks = []
    start = 0

    while start < len(cleaned):
        end = start + KNOWLEDGE_BASE_CHUNK_SIZE
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(cleaned):
            break
        start += step

    return chunks


def _split_documents(documents: list[Document]) -> list[Document]:
    chunks = []
    for document in documents:
        for chunk_index, chunk in enumerate(_chunk_text(document.page_content)):
            metadata = document.metadata.copy()
            metadata["chunk_index"] = chunk_index
            chunks.append(Document(page_content=chunk, metadata=metadata))
    return chunks


def _reset_vectorstore_dir():
    persist_path = Path(CHROMA_PERSIST_DIR)
    if persist_path.exists():
        shutil.rmtree(persist_path)
    persist_path.mkdir(parents=True, exist_ok=True)


def _write_manifest(manifest: dict):
    INGESTION_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def build_vectorstore(force_rebuild: bool = False):
    manifest = _knowledge_base_manifest()
    if not manifest["files"]:
        raise FileNotFoundError(
            f"No supported knowledge-base files were found in {Path(KNOWLEDGE_BASE_DIR).resolve()}"
        )

    if not force_rebuild and _manifest_is_current():
        return load_vectorstore()

    documents = _load_knowledge_documents()
    chunks = _split_documents(documents)

    _reset_vectorstore_dir()
    db = Chroma.from_documents(
        chunks,
        get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )
    _write_manifest(manifest)
    print(
        f"Vector store built successfully from {len(documents)} documents into {len(chunks)} chunks."
    )
    return db


def load_vectorstore():
    persist_path = Path(CHROMA_PERSIST_DIR)
    if not persist_path.exists() or not _manifest_is_current():
        build_vectorstore(force_rebuild=True)

    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=get_embeddings(),
    )


def add_to_vectorstore(text: str, metadata: dict | None = None):
    db = load_vectorstore()
    metadata = metadata or {}

    chunks = _chunk_text(text)
    metadatas = [metadata.copy() for _ in chunks]
    db.add_texts(chunks, metadatas=metadatas)
