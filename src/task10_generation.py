"""
Task 10 — Generation có citation.

Luồng: retrieve -> gán nhãn citation -> reorder chống lost-in-the-middle ->
format context -> gọi LLM provider -> trả GenerationResult.

Một chi tiết dễ sai: nhãn `[Document N]` phải được gán TRƯỚC khi reorder.
Nếu đánh số theo vị trí sau khi reorder thì `[Document 2]` trong câu trả lời
sẽ trỏ vào chunk khác với `sources[1]`, và citation không còn đối chiếu được
— trong khi `sources` bắt buộc phải sort theo score giảm dần.

Không đủ evidence hoặc provider lỗi thì trả safe refusal; không bịa thông tin.
"""

import os

from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve


load_dotenv()

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3
MAX_TOKENS = 1024

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "")

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "anthropic": "claude-sonnet-4-5",
}

REFUSAL_MESSAGE = (
    "Tôi không thể xác minh thông tin này từ nguồn hiện có. "
    "Bộ tài liệu đang dùng chỉ gồm quy chế đào tạo, quyết định học phí và "
    "thông báo dịch vụ sinh viên của ĐH Bách khoa Hà Nội."
)

SYSTEM_PROMPT = """Bạn là trợ lý giải đáp về dịch vụ sinh viên của Đại học Bách khoa Hà Nội.

Quy tắc bắt buộc:
- Chỉ trả lời dựa trên context được cung cấp. Không dùng kiến thức bên ngoài.
- Mỗi khẳng định phải kèm citation dạng [Document N] đúng với số hiệu trong context.
- Khi context không chứa đủ thông tin, trả lời đúng một câu: "{refusal}"
- Không suy đoán con số, thời hạn hay điều kiện không có trong context.
- Trả lời bằng tiếng Việt, ngắn gọn, đi thẳng vào câu hỏi.""".format(
    refusal=REFUSAL_MESSAGE
)


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Đưa chunks quan trọng về đầu và cuối context (chống lost-in-the-middle).

    Không mutate input: trả về list mới chứa chính các phần tử đã nhận.
    """
    if len(chunks) <= 2:
        return list(chunks)
    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]


def format_context(chunks: list[dict]) -> str:
    """Tạo context có số hiệu, title và source để citation kiểm chứng được."""
    parts: list[str] = []
    for position, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        label = chunk.get("citation_index", position)
        parts.append(
            f"[Document {label} | Title: {metadata.get('title', 'N/A')} | "
            f"Source: {metadata.get('source', 'N/A')}]\n{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


def call_llm(system_prompt: str, user_message: str) -> str:
    """Gọi OpenAI, Gemini hoặc Anthropic theo cấu hình .env."""
    provider = LLM_PROVIDER.lower()
    model = LLM_MODEL or DEFAULT_MODELS.get(provider, "")

    if provider == "openai":
        from openai import OpenAI

        response = OpenAI().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
        )
        return (response.choices[0].message.content or "").strip()

    if provider == "gemini":
        from google import genai
        from google.genai import types

        response = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY")
        ).models.generate_content(
            model=model,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_output_tokens=MAX_TOKENS,
            ),
        )
        return (response.text or "").strip()

    if provider == "anthropic":
        import anthropic

        response = anthropic.Anthropic().messages.create(
            model=model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
        )
        return "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()

    raise ValueError(f"LLM_PROVIDER không hỗ trợ: {LLM_PROVIDER}")


def build_prompt(query: str, chunks: list[dict], history: str = "") -> str:
    """Ghép context + lịch sử hội thoại (nếu có) thành user message."""
    labelled = [
        {**chunk, "citation_index": index}
        for index, chunk in enumerate(chunks, 1)
    ]
    context = format_context(reorder_for_llm(labelled))
    prefix = f"Lịch sử hội thoại gần đây:\n{history}\n\n" if history else ""
    return f"{prefix}Context:\n{context}\n\nCâu hỏi: {query}"


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Trả về GenerationResult: answer, sources và retrieval_source."""
    chunks = retrieve(query, top_k=top_k)
    if not chunks:
        return {
            "answer": REFUSAL_MESSAGE,
            "sources": [],
            "retrieval_source": "none",
        }

    retrieval_source = (
        "pageindex" if chunks[0]["retrieval_method"] == "pageindex" else "hybrid"
    )

    try:
        answer = call_llm(SYSTEM_PROMPT, build_prompt(query, chunks))
    except Exception as error:  # provider lỗi -> safe refusal, không crash UI
        print(f"LLM provider error: {error}")
        return {
            "answer": REFUSAL_MESSAGE,
            "sources": chunks,
            "retrieval_source": retrieval_source,
        }

    if not answer:
        answer = REFUSAL_MESSAGE

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": retrieval_source,
    }


if __name__ == "__main__":
    result = generate_with_citation("Học bổng khuyến khích học tập có mấy mức?")
    print(result["answer"])
    print(f"\nretrieval_source = {result['retrieval_source']}")
    for index, source in enumerate(result["sources"], 1):
        print(f"[{index}] {source['metadata']['title']} — {source['id']}")
