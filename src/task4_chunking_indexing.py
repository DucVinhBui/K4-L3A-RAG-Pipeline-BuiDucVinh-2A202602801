"""
Task 4 — Chunking, embedding và indexing.

Luồng: đọc Markdown đã chuẩn hóa (kèm front matter từ Task 3) -> chunk bằng
RecursiveCharacterTextSplitter -> embed bằng đúng một provider -> upsert vào
ChromaDB với cosine distance.

ID chunk = "<đường dẫn tương đối>::chunk-<index>" nên ổn định giữa các lần
chạy: `upsert` ghi đè đúng bản ghi cũ thay vì nhân bản dữ liệu.

Task 5 import lại `embed_texts()` từ đây để query và corpus luôn cùng model,
cùng dimension.
"""

import os
import re
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Giải thích lựa chọn tham số trong báo cáo nhóm.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or "BAAI/bge-m3"
EMBEDDING_DIM = 1024

COLLECTION_NAME = "rag_documents"

EMBED_BATCH_SIZE = 32

_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)


@lru_cache(maxsize=1)
def _sentence_transformer():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed danh sách text bằng provider trong .env (một provider duy nhất)."""
    if not texts:
        return []

    provider = EMBEDDING_PROVIDER.lower()

    if provider == "sentence_transformers":
        model = _sentence_transformer()
        vectors = model.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=len(texts) > EMBED_BATCH_SIZE,
            normalize_embeddings=True,
        )
        return [vector.tolist() for vector in vectors]

    if provider == "openai":
        from openai import OpenAI

        client = OpenAI()
        model = EMBEDDING_MODEL or "text-embedding-3-small"
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 64):
            batch = texts[start:start + 64]
            response = client.embeddings.create(model=model, input=batch)
            vectors.extend(item.embedding for item in response.data)
        return vectors

    if provider == "gemini":
        from google import genai

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        model = EMBEDDING_MODEL or "text-embedding-004"
        response = client.models.embed_content(model=model, contents=texts)
        return [list(item.values) for item in response.embeddings]

    raise ValueError(f"EMBEDDING_PROVIDER không hỗ trợ: {EMBEDDING_PROVIDER}")


def get_collection():
    """Mở Chroma collection dùng cosine distance."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _parse_front_matter(text: str) -> tuple[dict, str]:
    """Tách front matter YAML đơn giản khỏi phần thân Markdown."""
    match = _FRONT_MATTER.match(text)
    if not match:
        return {}, text

    fields: dict[str, str | None] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, _, raw = line.partition(":")
        value = raw.strip()
        if value in {"null", "~", ""}:
            fields[key.strip()] = None
            continue
        if len(value) >= 2 and value[0] == value[-1] == '"':
            value = value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        fields[key.strip()] = value
    return fields, text[match.end():]


def load_documents() -> list[dict]:
    """Đọc Markdown đã chuẩn hóa và trả về danh sách Document theo contract."""
    documents: list[dict] = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        raw = path.read_text(encoding="utf-8")
        front_matter, body = _parse_front_matter(raw)
        body = body.strip()
        if not body:
            continue

        doc_type = front_matter.get("doc_type") or (
            "legal" if "legal" in path.parts else "news"
        )
        documents.append(
            {
                "id": path.relative_to(STANDARDIZED_DIR).as_posix(),
                "content": body,
                "metadata": {
                    "source": front_matter.get("source") or path.name,
                    "title": front_matter.get("title") or path.stem,
                    "doc_type": doc_type,
                    "url": front_matter.get("url"),
                },
            }
        )
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id ổn định và chunk_index."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict] = []
    for document in documents:
        index = 0
        for text in splitter.split_text(document["content"]):
            text = text.strip()
            if not text:
                continue
            chunks.append(
                {
                    "id": f"{document['id']}::chunk-{index}",
                    "content": text,
                    "metadata": {**document["metadata"], "chunk_index": index},
                }
            )
            index += 1
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vào từng chunk, giữ nguyên các field sẵn có."""
    vectors = embed_texts([chunk["content"] for chunk in chunks])
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    return chunks


def _chroma_safe(metadata: dict) -> dict:
    """Chroma không nhận None trong metadata -> thay bằng chuỗi rỗng."""
    return {
        key: ("" if value is None else value) for key, value in metadata.items()
    }


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB (chạy lại không tạo bản ghi trùng)."""
    if not chunks:
        print("No chunks to index")
        return

    collection = get_collection()
    for start in range(0, len(chunks), 256):
        batch = chunks[start:start + 256]
        collection.upsert(
            ids=[chunk["id"] for chunk in batch],
            documents=[chunk["content"] for chunk in batch],
            embeddings=[chunk["embedding"] for chunk in batch],
            metadatas=[_chroma_safe(chunk["metadata"]) for chunk in batch],
        )


def run_pipeline() -> None:
    """Chạy load, chunk, embed và index."""
    documents = load_documents()
    print(f"Loaded {len(documents)} documents")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    embedded_chunks = embed_chunks(chunks)
    index_to_vectorstore(embedded_chunks)
    print(f"Indexed {len(embedded_chunks)} chunks into {COLLECTION_NAME}")
    print(f"Collection size: {get_collection().count()}")


if __name__ == "__main__":
    run_pipeline()
