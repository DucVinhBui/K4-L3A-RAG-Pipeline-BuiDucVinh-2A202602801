"""
Bonus — Cross-encoder reranker, so sánh trực tiếp với RRF.

Khác biệt bản chất so với RRF: RRF chỉ nhìn *thứ hạng*, không nhìn nội
dung. Nó không biết chunk hạng 1 của BM25 có thật sự trả lời câu hỏi hay
chỉ trùng nhiều từ khoá. Đó đúng là điểm hỏng đã đo được trên corpus này —
BM25 xếp hạng 1 cho bản dịch tiếng Anh gần trùng và RRF tin theo.

Cross-encoder đọc cặp (query, chunk) cùng lúc qua một transformer duy nhất
nên bắt được quan hệ ngữ nghĩa mà bi-encoder bỏ lỡ. Đánh đổi: phải chạy
model trên từng cặp, không cache được vector như dense search.

Dùng BAAI/bge-reranker-v2-m3 cho cùng họ với embedding BAAI/bge-m3 và có
hỗ trợ tiếng Việt. Model trả logit; đưa qua sigmoid để score về [0, 1] cho
so sánh được với cosine score ở các nhánh khác.

Kết quả vẫn mang `retrieval_method="hybrid"` để đúng contract, kèm khoá phụ
`rerank_method="cross_encoder"` cho phần hiển thị và phân tích.
"""

import math
import os

from dotenv import load_dotenv


load_dotenv()

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
MAX_LENGTH = 512

_model = None


def get_reranker():
    """Nạp cross-encoder một lần rồi giữ lại (model nặng ~2.2GB)."""
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(RERANKER_MODEL, max_length=MAX_LENGTH)
    return _model


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """Chấm lại từng candidate bằng cross-encoder và sort theo score giảm dần."""
    if not query.strip() or not candidates or top_k <= 0:
        return []

    pairs = [(query, candidate["content"]) for candidate in candidates]
    logits = get_reranker().predict(pairs)

    reranked: list[dict] = []
    for candidate, logit in zip(candidates, logits):
        item = dict(candidate)
        item["metadata"] = dict(candidate["metadata"])
        item["score"] = _sigmoid(float(logit))
        # Giữ "hybrid": docs/MODULE_CONTRACTS.md chỉ cho phép dense|bm25|hybrid|
        # pageindex, và kết quả này đúng là fusion của pool dense + BM25 — chỉ
        # khác ở cách chấm. Thêm "cross_encoder" vào contract để tiện hiển thị
        # là sửa contract của bài cho hợp code, ngược thứ tự.
        item["retrieval_method"] = "hybrid"
        item["rerank_method"] = "cross_encoder"
        reranked.append(item)

    reranked.sort(key=lambda item: item["score"], reverse=True)
    return reranked[:top_k]


def merge_candidates(ranked_lists: list[list[dict]]) -> list[dict]:
    """Gộp nhiều ranked list thành pool ứng viên, giữ bản gặp đầu tiên của mỗi id.

    Cross-encoder chấm lại từ đầu nên score cũ không còn ý nghĩa; chỉ cần
    đảm bảo mỗi chunk xuất hiện đúng một lần trong pool.
    """
    pool: list[dict] = []
    seen: set[str] = set()
    for ranked in ranked_lists:
        for item in ranked:
            if item["id"] in seen:
                continue
            seen.add(item["id"])
            pool.append(item)
    return pool


if __name__ == "__main__":
    from .task5_semantic_search import semantic_search
    from .task6_lexical_search import lexical_search

    question = "Điều kiện để sinh viên được xét công nhận tốt nghiệp là gì?"
    pool = merge_candidates(
        [semantic_search(question, top_k=10), lexical_search(question, top_k=10)]
    )
    print(f"{len(pool)} ứng viên -> cross-encoder\n")
    for result in rerank_cross_encoder(question, pool, top_k=3):
        print(f"{result['score']:.5f}  {result['metadata']['title']}")
        print(f"         {result['content'][:110]}...")
