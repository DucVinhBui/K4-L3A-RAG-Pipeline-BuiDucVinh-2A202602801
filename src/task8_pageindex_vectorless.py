"""
Task 8 — PageIndex vectorless fallback.

Khi dense retrieval không đủ tự tin (cosine score thấp), pipeline thử một
đường truy hồi khác hẳn: PageIndex duyệt cây mục lục của văn bản gốc thay vì
so khớp vector, nên bắt được câu hỏi hỏi theo cấu trúc ("Điều nào quy định
về cảnh báo học tập?") mà chunk embedding dễ bỏ sót.

Phạm vi: chỉ upload PDF chính sách trong data/landing/legal. PageIndex nhận
PDF, còn news đã ở dạng Markdown ngắn nên hybrid retrieval xử lý tốt rồi.

PageIndex là dịch vụ ngoài. Mọi lỗi (thiếu key, timeout, đổi schema) đều ném
ra ngoài để Task 9 bắt và quay về hybrid result — UI không bao giờ crash vì
provider này.
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
LEGAL_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
DOC_ID_CACHE = Path(__file__).parent.parent / "pageindex_doc_ids.json"

RETRIEVAL_TIMEOUT_SECONDS = 45
POLL_INTERVAL_SECONDS = 2


def _client():
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("PAGEINDEX_API_KEY chưa được cấu hình")
    from pageindex import PageIndexClient

    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def _load_cache() -> dict[str, str]:
    if DOC_ID_CACHE.exists():
        return json.loads(DOC_ID_CACHE.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict[str, str]) -> None:
    DOC_ID_CACHE.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _first(payload: dict, *keys, default=None):
    """SDK trả tên field khác nhau giữa các version — thử lần lượt."""
    for key in keys:
        if isinstance(payload, dict) and payload.get(key) is not None:
            return payload[key]
    return default


def upload_documents() -> dict[str, str]:
    """Upload PDF chính sách, cache document IDs để không upload lại."""
    client = _client()
    cache = _load_cache()

    for path in sorted(LEGAL_DIR.glob("*.pdf")):
        if path.name in cache:
            print(f"Skip (cached): {path.name} -> {cache[path.name]}")
            continue
        response = client.submit_document(file_path=str(path))
        doc_id = _first(response, "doc_id", "id", "documentId")
        if not doc_id:
            print(f"Failed: {path.name} — không lấy được doc_id: {response}")
            continue
        cache[path.name] = doc_id
        print(f"Uploaded: {path.name} -> {doc_id}")

    _save_cache(cache)
    return cache


def _wait_for_retrieval(client, retrieval_id: str) -> dict:
    deadline = time.time() + RETRIEVAL_TIMEOUT_SECONDS
    while time.time() < deadline:
        payload = client.get_retrieval(retrieval_id)
        status = str(_first(payload, "status", default="")).lower()
        if status in {"completed", "success", "done", ""}:
            return payload
        if status in {"failed", "error"}:
            raise RuntimeError(f"PageIndex retrieval failed: {payload}")
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError("PageIndex retrieval quá thời gian chờ")


def _extract_nodes(payload: dict) -> list[dict]:
    container = _first(payload, "retrieval", "result", "data", default=payload)
    if isinstance(container, dict):
        nodes = _first(
            container, "retrieved_nodes", "nodes", "results", "sources", default=[]
        )
    else:
        nodes = container
    return [node for node in (nodes or []) if isinstance(node, dict)]


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Trả về pageindex SearchResult (score giảm dần theo thứ hạng)."""
    if not query.strip() or top_k <= 0:
        return []

    client = _client()
    cache = _load_cache()
    if not cache:
        raise RuntimeError("Chưa có document nào được upload lên PageIndex")

    collected: list[dict] = []
    for source_name, doc_id in cache.items():
        submission = client.submit_query(doc_id=doc_id, query=query)
        retrieval_id = _first(submission, "retrieval_id", "id")
        payload = (
            _wait_for_retrieval(client, retrieval_id) if retrieval_id else submission
        )
        for node in _extract_nodes(payload):
            content = str(
                _first(node, "text", "content", "node_text", "summary", default="")
            ).strip()
            if not content:
                continue
            collected.append(
                {
                    "content": content,
                    "source": source_name,
                    "node_id": str(_first(node, "node_id", "id", default=len(collected))),
                    "raw_score": _first(node, "score", "relevance", default=None),
                }
            )

    # API không đảm bảo có score -> xếp hạng theo raw_score nếu có, và luôn
    # quy về một thang giảm dần theo rank để tuân thủ contract.
    collected.sort(
        key=lambda item: (
            item["raw_score"] if isinstance(item["raw_score"], (int, float)) else 0.0
        ),
        reverse=True,
    )

    results: list[dict] = []
    for rank, item in enumerate(collected[:top_k]):
        results.append(
            {
                "id": f"pageindex::{item['source']}::{item['node_id']}",
                "content": item["content"],
                "score": 1.0 / (1 + rank),
                "metadata": {
                    "source": item["source"],
                    "title": Path(item["source"]).stem,
                    "doc_type": "legal",
                    "url": None,
                    "chunk_index": rank,
                },
                "retrieval_method": "pageindex",
            }
        )
    return results


if __name__ == "__main__":
    upload_documents()
