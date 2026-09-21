"""
Bonus — Conversation memory: viết lại câu hỏi follow-up thành câu độc lập.

Giả thuyết ban đầu của tôi sai, và phép đo cho thấy sai ở đâu. Tôi cho rằng
follow-up hỏng vì retrieval mù: câu "Mức cao nhất bằng bao nhiêu lần mức
thấp nhất?" không còn từ khoá nào để bấu víu. Nhưng
`scripts/evaluate_memory.py` đo được context hit-rate **1.000** khi không có
memory — retrieval vẫn lấy đúng chunk. Chỗ hỏng nằm ở generation: model có
đủ context trong tay nhưng không nối được "mức cao nhất" với học bổng, nên
trả về safe refusal.

Vì vậy ghép lịch sử vào prompt đã đủ sửa phần lớn: accuracy 0.714 -> 0.857
trên bộ 7 hội thoại.

Viết lại query bằng lịch sử là bước thêm, và trên corpus này nó **không có
lợi**: 0.714, bằng đúng nhánh không memory. Nó sửa được fu-01 (model chịu
trả lời khi câu hỏi tự đủ nghĩa) nhưng làm hỏng fu-07 — bản viết lại
"Điểm trung bình tích lũy của sinh viên phải đạt bao nhiêu?" đánh rơi cụm
"để được xét công nhận tốt nghiệp", retrieval mất luôn chunk đúng và câu
trả lời thành refusal. Đổi một lỗi lấy một lỗi nặng hơn: hỏng ở prompt thì
context vẫn còn đó, hỏng ở retrieval thì không còn gì để cứu.

Nên hàm này giữ ở dạng tuỳ chọn, mặc định tắt trong app.py. n=7 là nhỏ,
chênh một case đã là 0.143, nên đây là căn cứ để chưa bật chứ chưa đủ để
kết luận rewriting vô dụng nói chung.

Chi phí: thêm 1 LLM call mỗi lượt follow-up (~0.94s đo được).
"""

REWRITE_SYSTEM_PROMPT = """Bạn viết lại câu hỏi của người dùng thành một câu hỏi độc lập, đọc riêng vẫn đủ nghĩa.

Quy tắc:
- Thay mọi đại từ hồi chỉ ("này", "đó", "họ") bằng danh từ cụ thể lấy từ lịch sử.
- Bổ sung phần bị lược. "Còn X?" phải thành một câu hỏi hoàn chỉnh về X.
- Giữ nguyên ý định hỏi. Không thêm điều kiện, không thu hẹp, không mở rộng.
- Không trả lời câu hỏi. Chỉ trả về đúng câu hỏi đã viết lại, không giải thích.
- Nếu câu hỏi vốn đã độc lập, trả về nguyên văn."""

MAX_REWRITE_CHARS = 300


def rewrite_followup(query: str, history: str) -> str:
    """Viết lại follow-up thành câu độc lập. Không có lịch sử hoặc lỗi thì giữ nguyên."""
    if not query.strip() or not history.strip():
        return query

    # Import trong hàm: task10 -> task9, tránh vòng phụ thuộc ở mức module.
    from .task10_generation import call_llm

    try:
        rewritten = call_llm(
            REWRITE_SYSTEM_PROMPT,
            f"Lịch sử hội thoại:\n{history}\n\nCâu hỏi cần viết lại: {query}",
        )
    except Exception as error:  # hỏng thì degrade về query gốc, không crash
        print(f"Query rewrite unavailable: {error}")
        return query

    rewritten = rewritten.strip().strip('"').strip()[:MAX_REWRITE_CHARS]
    return rewritten or query


if __name__ == "__main__":
    demo_history = (
        "Người dùng: Học bổng Trần Đại Nghĩa dành cho đối tượng nào?\n"
        "Trợ lý: Dành cho sinh viên có hoàn cảnh kinh tế đặc biệt khó khăn "
        "nhưng kết quả học tập tốt."
    )
    for question in ("Học bổng này có mấy mức?", "Còn cổng Trần Đại Nghĩa?"):
        print(f"gốc:       {question}")
        print(f"viết lại:  {rewrite_followup(question, demo_history)}\n")
