"""Đo định lượng conversation memory trên câu hỏi follow-up.

Bonus "conversation memory" chỉ được tính khi có kết quả đo, không phải khi
tính năng chạy được. Script chạy cùng một bộ hội thoại hai lượt qua ba nhánh
khác nhau đúng ở chỗ lịch sử được dùng ở đâu:

    none    — không lịch sử. Retrieve bằng lượt 2 nguyên văn, prompt không
              có lịch sử.
    prompt  — lịch sử chỉ ghép vào prompt. Retrieve vẫn bằng lượt 2 nguyên
              văn. Đây là cách làm hiển nhiên và là hành vi cũ của app.py.
    rewrite — lịch sử dùng để viết lại lượt 2 thành câu độc lập TRƯỚC khi
              retrieve, rồi vẫn ghép vào prompt.

Tách ba nhánh vì nhánh `prompt` cho kết quả gần như bằng `none`: lịch sử
đến quá muộn. Câu follow-up bị lược chủ ngữ thì retrieval không có gì để
bám, và prompt có đủ lịch sử cũng không cứu được context đã lấy sai.
Chỉ số `context_hit` đo đúng chỗ đó, tách phần hỏng ở retrieval ra khỏi
phần hỏng ở generation.

Thước đo tất định, không dùng LLM-judge: câu trả lời có chứa dữ kiện đúng
không (`expected_any`), context truy hồi có chứa dữ kiện đó không, và câu
trả lời có rơi vào safe refusal không. Với câu hỏi chỉ cần kiểm tra một con
số thì judge chỉ thêm nhiễu.

Mọi câu lượt 2 đều là ellipsis hoặc có đại từ hồi chỉ — đọc riêng thì không
đủ nghĩa. Đó là điều kiện để phép đo nói lên được điều gì.

Cách chạy:
    python -m scripts.evaluate_memory
"""

import json
import re
import time
import unicodedata
from pathlib import Path

from dotenv import load_dotenv

from src.bonus_conversation_memory import rewrite_followup
from src.task9_retrieval_pipeline import run_retrieval
from src.task10_generation import (
    LLM_MODEL,
    LLM_PROVIDER,
    REFUSAL_MESSAGE,
    SYSTEM_PROMPT,
    build_prompt,
    call_llm,
)


load_dotenv()

EVALUATION_DIR = Path(__file__).parent.parent / "group_project" / "evaluation"
FOLLOWUP_DATASET = EVALUATION_DIR / "followup_dataset.json"
RESULTS_JSON = EVALUATION_DIR / "memory_results.json"

TOP_K = 5
REFUSAL_MARKER = "không thể xác minh"

VARIANTS = {
    "none": {"label": "Không memory", "history": False, "rewrite": False},
    "prompt": {"label": "Memory trong prompt", "history": True, "rewrite": False},
    "rewrite": {
        "label": "Memory + rewrite query",
        "history": True,
        "rewrite": True,
    },
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text.lower())).strip()


def contains_fact(text: str, expected_any: list[str]) -> bool:
    haystack = normalize(text)
    return any(normalize(candidate) in haystack for candidate in expected_any)


def is_refusal(answer: str) -> bool:
    return REFUSAL_MARKER in normalize(answer)


def format_turn_1(question: str, answer: str) -> str:
    """Dựng history đúng định dạng app.py dùng, để phép đo khớp sản phẩm thật."""
    return f"Người dùng: {question[:400]}\nTrợ lý: {answer[:400]}"


def answer_turn(question: str, history: str = "") -> tuple[str, list[str], float]:
    chunks = run_retrieval(question, top_k=TOP_K)
    contexts = [chunk["content"] for chunk in chunks]
    if not chunks:
        return REFUSAL_MESSAGE, contexts, 0.0
    started = time.time()
    try:
        response = call_llm(SYSTEM_PROMPT, build_prompt(question, chunks, history))
    except Exception as error:
        print(f"    generation failed: {error}")
        return REFUSAL_MESSAGE, contexts, time.time() - started
    return response, contexts, time.time() - started


def main() -> None:
    cases = json.loads(FOLLOWUP_DATASET.read_text(encoding="utf-8"))
    print(f"Follow-up dataset: {len(cases)} hội thoại hai lượt\n")
    print("Warm-up retrieval...\n")
    run_retrieval("warm up", top_k=TOP_K)

    rows: list[dict] = []
    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {case['id']}")
        print(f"  lượt 1: {case['turn_1']}")
        # Lượt 1 giống hệt nhau ở ba nhánh: chạy một lần, dùng chung.
        turn_1_answer, _, _ = answer_turn(case["turn_1"])
        history = format_turn_1(case["turn_1"], turn_1_answer)
        print(f"  lượt 2: {case['turn_2']}")

        row = {
            "id": case["id"],
            "category": case["category"],
            "turn_1": case["turn_1"],
            "turn_1_answer": turn_1_answer,
            "turn_2": case["turn_2"],
            "expected_any": case["expected_any"],
            "why_needs_history": case["why_needs_history"],
            "variants": {},
        }

        for name, variant in VARIANTS.items():
            rewrite_s = 0.0
            question = case["turn_2"]
            if variant["rewrite"]:
                started = time.time()
                question = rewrite_followup(case["turn_2"], history)
                rewrite_s = time.time() - started

            answer, contexts, generation_s = answer_turn(
                question, history if variant["history"] else ""
            )
            row["variants"][name] = {
                "label": variant["label"],
                "query_used": question,
                "answer": answer,
                "correct": contains_fact(answer, case["expected_any"]),
                "context_hit": any(
                    contains_fact(context, case["expected_any"])
                    for context in contexts
                ),
                "refused": is_refusal(answer),
                "rewrite_s": round(rewrite_s, 3),
                "generation_s": round(generation_s, 3),
            }
            result = row["variants"][name]
            print(
                f"    {variant['label']:<24} correct={str(result['correct']):<5} "
                f"context_hit={str(result['context_hit']):<5} "
                f"refused={result['refused']}"
            )
            if variant["rewrite"] and question != case["turn_2"]:
                print(f"      viết lại: {question}")
        print()
        rows.append(row)

    total = len(rows)
    summary = {
        name: {
            "label": variant["label"],
            "accuracy": sum(row["variants"][name]["correct"] for row in rows) / total,
            "context_hit_rate": sum(
                row["variants"][name]["context_hit"] for row in rows
            )
            / total,
            "refusal_rate": sum(row["variants"][name]["refused"] for row in rows)
            / total,
            "mean_rewrite_s": sum(row["variants"][name]["rewrite_s"] for row in rows)
            / total,
            "mean_generation_s": sum(
                row["variants"][name]["generation_s"] for row in rows
            )
            / total,
        }
        for name, variant in VARIANTS.items()
    }

    report = {
        "run": {
            "generator_model": f"{LLM_PROVIDER}/{LLM_MODEL}",
            "top_k": TOP_K,
            "dataset_size": total,
            "history_turns": 1,
        },
        "summary": summary,
        "rows": rows,
    }
    RESULTS_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved: {RESULTS_JSON}\n")

    names = list(VARIANTS)
    print("| Metric | " + " | ".join(summary[name]["label"] for name in names) + " |")
    print("| --- |" + " ---: |" * len(names))
    for key, label, fmt in (
        ("accuracy", "Follow-up accuracy", "{:.3f}"),
        ("context_hit_rate", "Context hit-rate", "{:.3f}"),
        ("refusal_rate", "Refusal rate", "{:.3f}"),
        ("mean_rewrite_s", "Mean rewrite (s)", "{:.2f}"),
        ("mean_generation_s", "Mean generation (s)", "{:.2f}"),
    ):
        cells = " | ".join(fmt.format(summary[name][key]) for name in names)
        print(f"| {label} | {cells} |")


if __name__ == "__main__":
    main()
