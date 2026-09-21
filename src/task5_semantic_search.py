"""
Task 5 — Semantic search.

Query được embed bằng chính `embed_texts()` của Task 4 nên corpus và query
luôn cùng model, cùng dimension. Chroma dùng cosine distance, đổi sang
similarity bằng `1 - distance`.

Score trả về ở đây là cosine score gốc — Task 9 dùng đúng score này để quyết
định fallback, không dùng RRF score.
"""

from .task4_chunking_indexing import embed_texts, get_collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về dense SearchResult theo score giảm dần."""
    if not query.strip() or top_k <= 0:
        return []

    query_vector = embed_texts([query])[0]
    response = get_collection().query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    results: list[dict] = []
    seen: set[str] = set()
    for item_id, content, metadata, distance in zip(
        response["ids"][0],
        response["documents"][0],
        response["metadatas"][0],
        response["distances"][0],
    ):
        if item_id in seen:
            continue
        seen.add(item_id)
        results.append(
            {
                "id": item_id,
                "content": content,
                "score": max(0.0, 1.0 - float(distance)),
                "metadata": dict(metadata),
                "retrieval_method": "dense",
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    for result in semantic_search("Học bổng khuyến khích học tập có mấy mức?", top_k=3):
        print(f"{result['score']:.4f}  {result['metadata']['title']}")
        print(f"        {result['content'][:120]}...")
