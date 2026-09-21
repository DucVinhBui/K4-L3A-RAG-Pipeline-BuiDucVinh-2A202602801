"""Contract tests cho ba module bonus.

Toàn bộ offline: không gọi API, không nạp cross-encoder ~2.2GB. Mọi ranh
giới ra ngoài (LLM provider, model reranker, vector store) đều được
monkeypatch — test phải chạy được trên máy chưa có key và chưa tải model.
"""

import pytest

from src.contracts import validate_search_results


def metadata(source: str = "tuition.md", chunk_index: int = 0) -> dict:
    return {
        "source": source,
        "title": "Tuition policy",
        "doc_type": "legal",
        "url": "https://university.example/tuition",
        "chunk_index": chunk_index,
    }


def result(item_id: str, score: float, method: str = "dense", content: str = "x") -> dict:
    return {
        "id": item_id,
        "content": content,
        "score": score,
        "metadata": metadata(chunk_index=int(item_id.rsplit("-", 1)[-1])),
        "retrieval_method": method,
    }


# --- cross-encoder ----------------------------------------------------------


def test_cross_encoder_sorts_by_score_and_tags_method(monkeypatch):
    from src import bonus_cross_encoder as module

    class FakeReranker:
        def predict(self, pairs):
            # logit cao cho chunk chứa "đúng"; đảo ngược thứ tự đầu vào
            return [5.0 if "đúng" in chunk else -5.0 for _, chunk in pairs]

    monkeypatch.setattr(module, "get_reranker", FakeReranker)

    candidates = [
        result("doc-0", 0.9, content="nội dung lạc đề"),
        result("doc-1", 0.1, content="nội dung đúng trọng tâm"),
    ]
    ranked = module.rerank_cross_encoder("câu hỏi", candidates, top_k=2)

    validate_search_results(ranked)
    assert [item["id"] for item in ranked] == ["doc-1", "doc-0"]
    # contract chỉ cho phép dense|bm25|hybrid|pageindex
    assert all(item["retrieval_method"] == "hybrid" for item in ranked)
    assert all(item["rerank_method"] == "cross_encoder" for item in ranked)
    assert ranked[0]["score"] > 0.5 > ranked[1]["score"]


def test_cross_encoder_does_not_mutate_input(monkeypatch):
    from src import bonus_cross_encoder as module

    class FakeReranker:
        def predict(self, pairs):
            return [1.0] * len(pairs)

    monkeypatch.setattr(module, "get_reranker", FakeReranker)

    candidates = [result("doc-0", 0.9)]
    module.rerank_cross_encoder("câu hỏi", candidates, top_k=1)

    assert candidates[0]["score"] == 0.9
    assert candidates[0]["retrieval_method"] == "dense"


def test_merge_candidates_dedups_across_lists():
    from src.bonus_cross_encoder import merge_candidates

    dense = [result("doc-0", 0.9), result("doc-1", 0.8)]
    sparse = [result("doc-1", 3.2, method="bm25"), result("doc-2", 1.1, method="bm25")]

    pool = merge_candidates([dense, sparse])
    assert [item["id"] for item in pool] == ["doc-0", "doc-1", "doc-2"]
    # giữ bản gặp đầu tiên: doc-1 đến từ dense
    assert pool[1]["retrieval_method"] == "dense"


def test_cross_encoder_handles_empty_input(monkeypatch):
    from src import bonus_cross_encoder as module

    monkeypatch.setattr(module, "get_reranker", lambda: pytest.fail("không được nạp model"))
    assert module.rerank_cross_encoder("câu hỏi", [], top_k=5) == []
    assert module.rerank_cross_encoder("  ", [result("doc-0", 0.5)], top_k=5) == []


# --- HyDE query expansion ---------------------------------------------------


def test_expand_query_returns_original_plus_hypothetical(monkeypatch):
    from src import bonus_query_expansion as module

    monkeypatch.setattr(module, "generate_hypothetical_document", lambda q: "đoạn giả định")
    assert module.expand_query("câu hỏi") == ["câu hỏi", "đoạn giả định"]


def test_expand_query_degrades_to_original_when_llm_fails(monkeypatch):
    from src import bonus_query_expansion as module

    monkeypatch.setattr(module, "generate_hypothetical_document", lambda q: "")
    assert module.expand_query("câu hỏi") == ["câu hỏi"]


def test_hyde_search_fuses_variants(monkeypatch):
    from src import bonus_query_expansion as module

    def fake_semantic_search(query, top_k=10):
        if query == "câu hỏi":
            return [result("doc-0", 0.7), result("doc-1", 0.6)]
        return [result("doc-1", 0.9), result("doc-2", 0.5)]

    monkeypatch.setattr(module, "semantic_search", fake_semantic_search)

    fused = module.hyde_search("câu hỏi", top_k=3, variants=["câu hỏi", "đoạn giả định"])
    validate_search_results(fused)
    # doc-1 có mặt ở cả hai list nên phải lên đầu sau RRF
    assert fused[0]["id"] == "doc-1"


# --- conversation memory ----------------------------------------------------


def test_rewrite_followup_keeps_query_without_history():
    from src.bonus_conversation_memory import rewrite_followup

    assert rewrite_followup("Còn cổng Trần Đại Nghĩa?", "") == "Còn cổng Trần Đại Nghĩa?"


def test_rewrite_followup_strips_quotes_from_model_output(monkeypatch):
    from src import task10_generation

    monkeypatch.setattr(
        task10_generation, "call_llm", lambda system, user: '  "Câu đã viết lại?"  '
    )
    from src.bonus_conversation_memory import rewrite_followup

    assert rewrite_followup("Còn X?", "lịch sử") == "Câu đã viết lại?"


def test_rewrite_followup_degrades_on_provider_error(monkeypatch):
    from src import task10_generation

    def boom(system, user):
        raise RuntimeError("provider down")

    monkeypatch.setattr(task10_generation, "call_llm", boom)
    from src.bonus_conversation_memory import rewrite_followup

    assert rewrite_followup("Còn X?", "lịch sử") == "Còn X?"


# --- pipeline routing -------------------------------------------------------


def test_run_retrieval_routes_to_cross_encoder(monkeypatch):
    from src import task9_retrieval_pipeline as pipeline

    monkeypatch.setattr(
        pipeline, "semantic_search", lambda query, top_k=10: [result("doc-0", 0.95)]
    )
    monkeypatch.setattr(
        pipeline,
        "lexical_search",
        lambda query, top_k=10: [result("doc-1", 2.0, method="bm25")],
    )

    from src import bonus_cross_encoder

    class FakeReranker:
        def predict(self, pairs):
            return [0.0] * len(pairs)

    monkeypatch.setattr(bonus_cross_encoder, "get_reranker", FakeReranker)

    results = pipeline.run_retrieval("câu hỏi", top_k=2, reranker="cross_encoder")
    validate_search_results(results)
    assert all(item["rerank_method"] == "cross_encoder" for item in results)


def test_run_retrieval_adds_ranked_list_per_variant(monkeypatch):
    from src import task9_retrieval_pipeline as pipeline

    seen: list[str] = []

    def fake_semantic_search(query, top_k=10):
        seen.append(query)
        return [result("doc-0", 0.95)]

    monkeypatch.setattr(pipeline, "semantic_search", fake_semantic_search)
    monkeypatch.setattr(pipeline, "lexical_search", lambda query, top_k=10: [])

    pipeline.run_retrieval(
        "câu hỏi", top_k=2, query_variants=["câu hỏi", "đoạn giả định"]
    )
    # query gốc chạy một lần, biến thể trùng query gốc không chạy lại
    assert seen == ["câu hỏi", "đoạn giả định"]


def test_run_retrieval_fallback_uses_original_query_score(monkeypatch):
    """Ngưỡng fallback phải so với cosine của query gốc, không phải của HyDE."""
    from src import task9_retrieval_pipeline as pipeline

    def fake_semantic_search(query, top_k=10):
        # query gốc yếu, biến thể HyDE mạnh
        return [result("doc-0", 0.2 if query == "câu hỏi" else 0.99)]

    monkeypatch.setattr(pipeline, "semantic_search", fake_semantic_search)
    monkeypatch.setattr(pipeline, "lexical_search", lambda query, top_k=10: [])

    called: list[str] = []

    def fake_pageindex(query, top_k=5):
        called.append(query)
        return []

    monkeypatch.setattr(pipeline, "pageindex_search", fake_pageindex)

    pipeline.run_retrieval(
        "câu hỏi",
        top_k=2,
        score_threshold=0.59,
        query_variants=["câu hỏi", "đoạn giả định"],
    )
    assert called == ["câu hỏi"], "fallback phải kích hoạt theo score của query gốc"
