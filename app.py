"""Chatbot RAG — Dịch vụ sinh viên Đại học Bách khoa Hà Nội."""

import time

import streamlit as st
from dotenv import load_dotenv

from src.task9_retrieval_pipeline import SCORE_THRESHOLD, retrieve
from src.task10_generation import (
    LLM_MODEL,
    LLM_PROVIDER,
    REFUSAL_MESSAGE,
    SYSTEM_PROMPT,
    build_prompt,
    call_llm,
)


load_dotenv()

st.set_page_config(
    page_title="RAG Chatbot — Dịch vụ sinh viên HUST",
    page_icon="🎓",
    layout="wide",
)

HISTORY_TURNS = 2

EXAMPLE_QUESTIONS = [
    "Học bổng khuyến khích học tập có mấy mức?",
    "Sinh viên bị cảnh báo học tập trong trường hợp nào?",
    "Học phí năm học 2025-2026 được tính như thế nào?",
    "Giá vé tàu Hà Nội – Sài Gòn là bao nhiêu?",  # ngoài domain
]


def format_history(messages: list[dict]) -> str:
    """Lấy vài lượt gần nhất để trả lời được câu hỏi follow-up."""
    turns = [m for m in messages if m["role"] in {"user", "assistant"}]
    recent = turns[-HISTORY_TURNS * 2:]
    if not recent:
        return ""
    return "\n".join(
        f"{'Người dùng' if m['role'] == 'user' else 'Trợ lý'}: {m['content'][:400]}"
        for m in recent
    )


def render_sources(sources: list[dict], retrieval_source: str, elapsed: float) -> None:
    """Hiển thị nguồn đã dùng, retrieval method và score."""
    if not sources:
        st.info("Không có nguồn nào vượt ngưỡng tin cậy cho câu hỏi này.")
        return

    st.caption(
        f"retrieval_source = `{retrieval_source}` · "
        f"{len(sources)} nguồn · {elapsed:.1f}s"
    )
    with st.expander(f"Nguồn đã dùng ({len(sources)})", expanded=False):
        for index, source in enumerate(sources, 1):
            metadata = source["metadata"]
            header = (
                f"**[Document {index}] {metadata['title']}** — "
                f"`{source['retrieval_method']}` · score `{source['score']:.4f}`"
            )
            st.markdown(header)
            url = metadata.get("url")
            location = f"{metadata['source']} · chunk {metadata['chunk_index']}"
            st.caption(f"[{location}]({url})" if url else location)
            st.markdown(
                f"> {source['content'][:500].replace(chr(10), ' ')}"
                f"{'…' if len(source['content']) > 500 else ''}"
            )
            if index < len(sources):
                st.divider()


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("🎓 RAG Chatbot")
    st.caption(
        "Hỏi đáp về học phí, học bổng, quy chế đào tạo và dịch vụ sinh viên "
        "của Đại học Bách khoa Hà Nội."
    )
    top_k = st.slider("Số chunks (top_k)", 3, 10, 5)
    use_reranking = st.toggle("Hybrid + RRF", value=True, help="Tắt = dense-only")
    use_memory = st.toggle("Nhớ hội thoại", value=True)
    threshold = st.slider(
        "Ngưỡng fallback (cosine)", 0.0, 1.0, float(SCORE_THRESHOLD), 0.05
    )
    st.divider()
    st.caption(f"LLM: `{LLM_PROVIDER}` · `{LLM_MODEL or 'default'}`")
    st.caption("Câu hỏi mẫu:")
    for question in EXAMPLE_QUESTIONS:
        st.caption(f"• {question}")
    if st.button("Xoá hội thoại", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("Hỏi đáp dịch vụ sinh viên HUST")
st.caption(
    "Câu trả lời chỉ dựa trên bộ tài liệu đã thu thập và luôn kèm citation "
    "`[Document N]` đối chiếu được với danh sách nguồn bên dưới mỗi câu trả lời."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(
                message.get("sources", []),
                message.get("retrieval_source", "none"),
                message.get("elapsed", 0.0),
            )

query = st.chat_input("Nhập câu hỏi...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        started = time.time()
        with st.spinner("Đang truy hồi và tổng hợp..."):
            history = (
                format_history(st.session_state.messages[:-1]) if use_memory else ""
            )
            sources = retrieve(
                query,
                top_k=top_k,
                score_threshold=threshold,
                use_reranking=use_reranking,
            )
            if not sources:
                answer = REFUSAL_MESSAGE
                retrieval_source = "none"
            else:
                retrieval_source = (
                    "pageindex"
                    if sources[0]["retrieval_method"] == "pageindex"
                    else "hybrid"
                )
                try:
                    answer = call_llm(
                        SYSTEM_PROMPT, build_prompt(query, sources, history)
                    )
                except Exception as error:
                    answer = f"{REFUSAL_MESSAGE}\n\n_(Lỗi provider: {error})_"

        elapsed = time.time() - started
        st.markdown(answer)
        render_sources(sources, retrieval_source, elapsed)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "retrieval_source": retrieval_source,
            "elapsed": elapsed,
        }
    )
