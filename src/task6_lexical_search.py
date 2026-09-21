"""
Task 6 — Lexical search bằng BM25.

Corpus lấy trực tiếp từ Chroma collection của Task 4 nên dense và BM25 luôn
xếp hạng trên đúng cùng một tập chunk và cùng một tập ID — điều kiện để RRF
ở Task 7 gộp được theo ID.

BM25 bù cho dense ở các truy vấn chứa mã văn bản, con số và tên riêng
("QĐ 5445", "Điều 22", "Trần Đại Nghĩa") — nơi embedding thường mờ nghĩa.

Vì sao BM25Plus chứ không phải BM25Okapi: idf của Okapi là
log(N - n + 0.5) - log(n + 0.5), bằng 0 khi term xuất hiện ở khoảng một nửa
số document và âm khi phổ biến hơn. Corpus ở đây nhỏ và rất đồng chủ đề
("sinh viên", "học phí" có mặt gần như khắp nơi) nên Okapi triệt tiêu đúng
những term đặc trưng của domain. BM25Plus dùng log((N+1)/n) nên idf luôn
dương và thứ hạng giữ được ý nghĩa.

Đổi lại, BM25Plus cộng một hằng số delta cho mọi document kể cả document
không chứa term nào, nên không thể lọc "liên quan" bằng score > 0. Ở đây lọc
bằng giao token: chunk phải chứa ít nhất một token của query.
"""

import re
import unicodedata


CORPUS: list[dict] = []

_TOKEN = re.compile(r"[0-9a-zà-ỹ]+", re.IGNORECASE)

_index_cache: dict[tuple, tuple[object, list[set[str]]]] = {}


def load_corpus() -> list[dict]:
    """Đọc toàn bộ chunks đã index để BM25 dùng chung corpus với dense."""
    from .task4_chunking_indexing import get_collection

    response = get_collection().get(include=["documents", "metadatas"])
    return [
        {"id": item_id, "content": content, "metadata": dict(metadata)}
        for item_id, content, metadata in zip(
            response["ids"], response["documents"], response["metadatas"]
        )
    ]


def _ensure_corpus() -> list[dict]:
    global CORPUS
    if not CORPUS:
        CORPUS = load_corpus()
    return CORPUS


def tokenize(text: str) -> list[str]:
    """Tokenizer đơn giản, giữ dấu tiếng Việt và tách số khỏi chữ."""
    normalized = unicodedata.normalize("NFC", text.lower())
    return _TOKEN.findall(normalized)


def build_bm25_index(corpus: list[dict]):
    """Tạo BM25 index từ cùng corpus chunks của Task 4."""
    from rank_bm25 import BM25Plus

    return BM25Plus([tokenize(item["content"]) for item in corpus])


def _cache_key(corpus: list[dict]) -> tuple:
    """Khóa cache nhận diện được corpus mà không phải tokenize lại toàn bộ.

    Không dùng riêng id(corpus): CPython tái sử dụng địa chỉ sau khi list cũ
    bị thu hồi, nên một corpus khác cùng độ dài có thể trúng cache cũ. Kèm
    thêm ID phần tử đầu/cuối là đủ phân biệt trong thực tế.
    """
    if not corpus:
        return (id(corpus), 0)
    return (id(corpus), len(corpus), corpus[0]["id"], corpus[-1]["id"])


def _cached_index(corpus: list[dict]) -> tuple[object, list[set[str]]]:
    """Build BM25 index một lần cho mỗi corpus và dùng lại giữa các query."""
    key = _cache_key(corpus)
    cached = _index_cache.get(key)
    if cached is None:
        cached = (
            build_bm25_index(corpus),
            [set(tokenize(item["content"])) for item in corpus],
        )
        _index_cache[key] = cached
    return cached


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về BM25 SearchResult theo score giảm dần."""
    corpus = _ensure_corpus()
    tokens = tokenize(query)
    if not corpus or not tokens or top_k <= 0:
        return []

    index, token_sets = _cached_index(corpus)
    scores = index.get_scores(tokens)
    query_tokens = set(tokens)

    candidates = [
        position
        for position in range(len(corpus))
        if query_tokens & token_sets[position]
    ]
    candidates.sort(key=lambda position: scores[position], reverse=True)

    results: list[dict] = []
    for position in candidates[:top_k]:
        item = corpus[position]
        results.append(
            {
                "id": item["id"],
                "content": item["content"],
                "score": float(scores[position]),
                "metadata": dict(item["metadata"]),
                "retrieval_method": "bm25",
            }
        )
    return results


if __name__ == "__main__":
    for result in lexical_search("học bổng Trần Đại Nghĩa", top_k=3):
        print(f"{result['score']:.4f}  {result['metadata']['title']}")
        print(f"        {result['content'][:120]}...")
