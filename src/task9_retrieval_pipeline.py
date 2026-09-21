"""
Task 9 — Retrieval pipeline hoàn chỉnh.

Luồng xử lý:
    1. Chạy semantic_search và lexical_search trên cùng corpus.
    2. Fuse hai danh sách bằng RRF đúng một lần.
    3. Lấy best cosine score gốc từ dense results (không phải RRF score).
    4. Nếu score dưới threshold, thử PageIndex fallback.
    5. Nếu fallback lỗi hoặc rỗng, trả hybrid results thay vì crash.

Vì sao dùng dense score chứ không phải RRF score: RRF score chỉ phụ thuộc
thứ hạng nên một query hoàn toàn ngoài domain vẫn cho RRF score y hệt một
query đúng domain (top-1 luôn được 1/61). Chỉ cosine score mới phản ánh
"corpus có thật sự chứa thông tin này không".

SCORE_THRESHOLD được hiệu chỉnh bằng script scripts/calibrate_threshold.py
trên tập query in-domain và out-of-domain — xem group_project/evaluation/RESULT.md.
"""

import os

from dotenv import load_dotenv

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


load_dotenv()

SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD") or 0.59)
DEFAULT_TOP_K = 5
CANDIDATE_MULTIPLIER = 2


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """Trả về hybrid hoặc pageindex SearchResult."""
    if not query.strip() or top_k <= 0:
        return []

    candidate_k = top_k * CANDIDATE_MULTIPLIER
    dense = semantic_search(query, top_k=candidate_k)

    if use_reranking:
        # Chỉ chạy BM25 khi thật sự fuse: nhánh dense-only mà vẫn tính BM25
        # rồi vứt đi thì phép so sánh A/B về chi phí không còn ý nghĩa.
        sparse = lexical_search(query, top_k=candidate_k)
        hybrid = rerank_rrf([dense, sparse], top_k=top_k)
    else:
        hybrid = dense[:top_k]

    best_dense_score = dense[0]["score"] if dense else 0.0
    if best_dense_score < score_threshold:
        try:
            fallback = pageindex_search(query, top_k=top_k)
            if fallback:
                return fallback[:top_k]
        except Exception as error:  # provider ngoài không được làm sập pipeline
            print(f"PageIndex fallback unavailable: {error}")

    return hybrid[:top_k]


if __name__ == "__main__":
    for result in retrieve("Điều kiện được xét học bổng khuyến khích học tập?", top_k=3):
        print(
            f"[{result['retrieval_method']}] {result['score']:.5f}  "
            f"{result['metadata']['title']}"
        )
