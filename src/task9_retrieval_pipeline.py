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

`retrieve()` giữ đúng chữ ký trong docs/MODULE_CONTRACTS.md và được test
contract khoá lại. Phần logic nằm ở `run_retrieval()` để các nhánh bonus
(src/bonus_pipeline.py) dùng lại cùng một cài đặt thay vì chép thành bản
thứ hai — hai bản cài đặt sẽ trôi khỏi nhau và phép so A/B mất ý nghĩa.
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


def run_retrieval(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
    reranker: str = "rrf",
    query_variants: list[str] | None = None,
) -> list[dict]:
    """Cài đặt đầy đủ; trả về hybrid, cross_encoder hoặc pageindex SearchResult.

    `query_variants` là các biến thể query đã sinh sẵn (HyDE). Chúng được
    truyền vào từ ngoài chứ không sinh ở đây, để chi phí LLM được bấm giờ
    riêng thay vì lẫn vào latency retrieval.

    Ngưỡng fallback luôn so với cosine của query GỐC. Để một đoạn văn do LLM
    bịa ra quyết định "corpus có chứa thông tin này không" là đưa ảo giác
    vào đúng cơ chế dựng lên để chống ảo giác.
    """
    if not query.strip() or top_k <= 0:
        return []

    candidate_k = top_k * CANDIDATE_MULTIPLIER

    # Luôn chạy dense trên query gốc: vừa là ranked list chính, vừa là tín
    # hiệu cosine duy nhất đáng tin để quyết định fallback.
    dense = semantic_search(query, top_k=candidate_k)
    best_dense_score = dense[0]["score"] if dense else 0.0
    ranked_lists = [dense]

    for variant in query_variants or []:
        if variant.strip() and variant != query:
            ranked_lists.append(semantic_search(variant, top_k=candidate_k))

    if use_reranking:
        # Chỉ chạy BM25 khi thật sự fuse: nhánh dense-only mà vẫn tính BM25
        # rồi vứt đi thì phép so sánh A/B về chi phí không còn ý nghĩa.
        ranked_lists.append(lexical_search(query, top_k=candidate_k))

    if reranker == "cross_encoder":
        from .bonus_cross_encoder import merge_candidates, rerank_cross_encoder

        candidates = merge_candidates(ranked_lists)
        results = rerank_cross_encoder(query, candidates, top_k=top_k)
    elif len(ranked_lists) > 1:
        results = rerank_rrf(ranked_lists, top_k=top_k)
    else:
        results = dense[:top_k]

    if best_dense_score < score_threshold:
        try:
            fallback = pageindex_search(query, top_k=top_k)
            if fallback:
                return fallback[:top_k]
        except Exception as error:  # provider ngoài không được làm sập pipeline
            print(f"PageIndex fallback unavailable: {error}")

    return results[:top_k]


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """Interface theo contract: dense + BM25, fuse bằng RRF, fallback PageIndex."""
    return run_retrieval(
        query,
        top_k=top_k,
        score_threshold=score_threshold,
        use_reranking=use_reranking,
    )


if __name__ == "__main__":
    for result in retrieve("Điều kiện được xét học bổng khuyến khích học tập?", top_k=3):
        print(
            f"[{result['retrieval_method']}] {result['score']:.5f}  "
            f"{result['metadata']['title']}"
        )
