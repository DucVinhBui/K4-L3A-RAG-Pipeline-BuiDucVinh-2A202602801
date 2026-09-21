"""
Bonus — HyDE (Hypothetical Document Embeddings) query expansion.

Vấn đề: câu hỏi và đoạn văn trả lời nó nằm ở hai vùng khác nhau của không
gian embedding. "Sinh viên bị cảnh báo học tập khi nào?" là một câu hỏi;
đoạn văn quy chế trả lời nó là một mệnh đề hành chính ("Nâng một mức cảnh
báo học tập đối với sinh viên có số TC không đạt..."). Cosine giữa hai thứ
đó thấp hơn cosine giữa hai đoạn văn hành chính cùng thể loại.

HyDE xử lý bằng cách cho LLM viết một đoạn văn giả định trả lời câu hỏi,
rồi tìm kiếm bằng embedding của đoạn văn đó thay vì của câu hỏi. Đoạn văn
giả định có thể sai sự thật — điều đó không quan trọng, nó chỉ cần đúng
*thể loại văn bản* để rơi vào đúng vùng embedding.

Tìm bằng cả câu hỏi gốc lẫn đoạn giả định rồi fuse bằng RRF: nếu LLM viết
lạc đề thì nhánh câu hỏi gốc vẫn giữ được kết quả đúng.

Chi phí: thêm đúng 1 LLM call mỗi query. Chi phí này được đo riêng
(`expansion_s`) trong scripts/evaluate.py, không gộp vào retrieval latency.
"""

from .task5_semantic_search import semantic_search
from .task7_reranking import rerank_rrf


HYDE_SYSTEM_PROMPT = """Bạn viết một đoạn trích giả định từ văn bản hành chính của Đại học Bách khoa Hà Nội.

Quy tắc:
- Viết đúng 2-3 câu, văn phong quy chế/thông báo, không phải văn nói.
- Viết như thể đang trích từ tài liệu gốc: nêu thẳng quy định, con số, điều kiện.
- Không mở đầu bằng "Theo tôi", "Câu trả lời là" hay bất kỳ lời dẫn nào.
- Không nói rằng bạn đang giả định. Chỉ trả về đoạn trích.
- Nếu không chắc con số cụ thể, vẫn viết một con số hợp lý — đoạn này chỉ
  dùng để tìm kiếm, không hiển thị cho người dùng."""

MAX_HYDE_CHARS = 600


def generate_hypothetical_document(query: str) -> str:
    """Sinh đoạn văn giả định trả lời câu hỏi. Lỗi provider trả về chuỗi rỗng."""
    if not query.strip():
        return ""

    # Import trong hàm: task10 import task9, task9 import module này.
    # Import ở mức module sẽ tạo vòng phụ thuộc.
    from .task10_generation import call_llm

    try:
        hypothetical = call_llm(HYDE_SYSTEM_PROMPT, f"Câu hỏi: {query}")
    except Exception as error:  # HyDE hỏng thì degrade về query gốc, không crash
        print(f"HyDE unavailable: {error}")
        return ""
    return hypothetical.strip()[:MAX_HYDE_CHARS]


def expand_query(query: str) -> list[str]:
    """Trả về các biến thể query để tìm kiếm; phần tử đầu luôn là query gốc."""
    hypothetical = generate_hypothetical_document(query)
    if not hypothetical:
        return [query]
    return [query, hypothetical]


def hyde_search(
    query: str, top_k: int = 10, variants: list[str] | None = None
) -> list[dict]:
    """Dense search trên từng biến thể query rồi fuse bằng RRF.

    `variants` cho phép caller sinh sẵn biến thể ở bước khác để đo riêng
    chi phí LLM; bỏ trống thì hàm tự gọi expand_query.
    """
    if not query.strip() or top_k <= 0:
        return []

    if variants is None:
        variants = expand_query(query)

    ranked_lists = [
        semantic_search(variant, top_k=top_k) for variant in variants if variant.strip()
    ]
    ranked_lists = [results for results in ranked_lists if results]
    if not ranked_lists:
        return []
    if len(ranked_lists) == 1:
        return ranked_lists[0][:top_k]

    return rerank_rrf(ranked_lists, top_k=top_k)


if __name__ == "__main__":
    question = "Sinh viên bị cảnh báo học tập trong trường hợp nào?"
    document = generate_hypothetical_document(question)
    print(f"Câu hỏi:      {question}")
    print(f"Đoạn giả định: {document}\n")
    for result in hyde_search(question, top_k=3, variants=[question, document]):
        print(f"{result['score']:.5f}  {result['metadata']['title']}")
        print(f"         {result['content'][:110]}...")
