"""
Task 7 — Reciprocal Rank Fusion.

Cosine score (0–1) và BM25 score (không chặn trên) khác thang đo nên không
cộng trực tiếp được. RRF chỉ dùng thứ hạng:

    RRF(d) = sum over các list of 1 / (k + rank),  rank bắt đầu từ 1

Lưu ý: RRF score chỉ phản ánh thứ hạng, không phải độ tương đồng, nên Task 9
không dùng nó để so với threshold fallback.
"""


def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
) -> list[dict]:
    """Fuse nhiều ranked lists theo ID và trả hybrid SearchResult."""
    if top_k <= 0:
        return []

    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        seen: set[str] = set()
        for rank, item in enumerate(ranked_list, 1):
            item_id = item["id"]
            if item_id in seen:  # một list không được cộng điểm hai lần
                continue
            seen.add(item_id)
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
            items.setdefault(item_id, item)

    ranked_ids = sorted(scores, key=lambda item_id: scores[item_id], reverse=True)

    results: list[dict] = []
    for item_id in ranked_ids[:top_k]:
        result = dict(items[item_id])
        result["metadata"] = dict(result["metadata"])
        result["score"] = scores[item_id]
        result["retrieval_method"] = "hybrid"
        results.append(result)
    return results


if __name__ == "__main__":
    from .task5_semantic_search import semantic_search
    from .task6_lexical_search import lexical_search

    question = "Mức học phí năm học 2025-2026 là bao nhiêu?"
    fused = rerank_rrf(
        [semantic_search(question, top_k=10), lexical_search(question, top_k=10)],
        top_k=5,
    )
    for result in fused:
        print(f"{result['score']:.5f}  {result['metadata']['title']}")
