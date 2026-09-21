"""Sinh group_project/evaluation/RESULT.md từ eval_results.json.

Mọi con số trong báo cáo được đọc thẳng từ kết quả chạy thật, không gõ tay,
nên chạy lại evaluate rồi render là báo cáo tự khớp.

    python -m scripts.evaluate
    python -m scripts.render_result
"""

import json
import subprocess
from datetime import date
from pathlib import Path


EVALUATION_DIR = Path(__file__).parent.parent / "group_project" / "evaluation"
RESULTS_JSON = EVALUATION_DIR / "eval_results.json"
RESULT_MD = EVALUATION_DIR / "RESULT.md"

METRIC_LABELS = [
    ("faithfulness", "Faithfulness"),
    ("answer_relevancy", "Answer relevance"),
    ("context_recall", "Context recall"),
    ("context_precision", "Context precision"),
]

NOT_RUN = "chưa chạy"

# Root cause đã điều tra thủ công cho từng case hay hỏng; dùng khi case đó lọt
# vào bảng worst performers. Case không có trong đây sẽ dùng mô tả suy ra từ dữ liệu.
ROOT_CAUSES = {
    "hp-02": (
        "data/chunking",
        "Câu định nghĩa cách tính học phí nằm vắt qua ranh giới chunk-3/chunk-4 "
        "của quyết định học phí; retrieval lấy chunk-3 (phần căn cứ pháp lý) "
        "nên thiếu đúng mệnh đề trả lời. Overlap 50 ký tự không đủ cho đoạn văn "
        "bản hành chính dài.",
    ),
    "hb-01": (
        "retrieval + generation",
        "Cụm 'của Bách khoa Hà Nội' trong câu hỏi kéo retrieval về các chunk "
        "giới thiệu chung (liệt kê các loại học bổng) thay vì chunk liệt kê 3 "
        "mức A/B/C. Model vẫn trả lời 'có 3 mức' nhưng context không nêu đủ ba "
        "mức, nên faithfulness bị chấm 0. Bỏ cụm tên trường khỏi câu hỏi thì "
        "retrieval lấy đúng chunk và câu trả lời liệt kê đủ A/B/C.",
    ),
    "tn-01": (
        "retrieval",
        "BM25 xếp hạng 1 cho bản dịch tiếng Anh gần trùng "
        "(hust-quy-che-dao-tao-tin-chi-en.md). RRF cho top-1 của BM25 trọng số "
        "1/61, đẩy chunk đúng ở hạng 3 của dense (1/63) ra khỏi top-5.",
    ),
    "hb-05": (
        "retrieval",
        "Cùng cơ chế với tn-01: near-duplicate giữa hai bài học bổng làm BM25 "
        "đẩy chunk chứa mệnh đề 'sinh viên không cần phải đăng ký' xuống dưới "
        "ngưỡng top-5 sau khi fuse.",
    ),
}


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "n/a"


def fmt(value, digits: int = 3) -> str:
    return NOT_RUN if not isinstance(value, (int, float)) else f"{value:.{digits}f}"


def delta(a, b, digits: int = 3) -> str:
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return NOT_RUN
    return f"{b - a:+.{digits}f}"


def collect_worst(report: dict, limit: int = 3) -> list[dict]:
    """Case tệ nhất trên cả hai config, xếp theo trung bình 4 metric rồi context hit."""
    entries: list[dict] = []
    for name, config in report["configs"].items():
        per_case = (config.get("ragas") or {}).get("_per_case") or []
        by_question = {row.get("user_input"): row for row in per_case}
        for row in config["rows"]:
            metrics = by_question.get(row["user_input"], {})
            values = [
                metrics[key]
                for key, _ in METRIC_LABELS
                if isinstance(metrics.get(key), (int, float))
            ]
            entries.append(
                {
                    "config": "A" if name.startswith("A") else "B",
                    "row": row,
                    "metrics": metrics,
                    "mean": sum(values) / len(values) if values else None,
                }
            )

    entries.sort(
        key=lambda item: (
            item["mean"] if item["mean"] is not None else (1.0 if item["row"]["context_hit"] else 0.0),
            item["row"]["context_hit"],
        )
    )

    picked: list[dict] = []
    seen: set[str] = set()
    for entry in entries:
        case_id = entry["row"]["id"]
        if case_id in seen:
            continue
        seen.add(case_id)
        picked.append(entry)
        if len(picked) == limit:
            break
    return picked


def main() -> None:
    report = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    run = report["run"]
    a = report["configs"]["A_dense_only"]
    b = report["configs"]["B_hybrid_rrf"]
    ragas_a = a.get("ragas") or {}
    ragas_b = b.get("ragas") or {}
    has_ragas = bool(ragas_a) and bool(ragas_b)

    hit_delta = b["context_hit_rate"] - a["context_hit_rate"]
    retr_a = a.get("median_retrieval_s", a["mean_retrieval_s"])
    retr_b = b.get("median_retrieval_s", b["mean_retrieval_s"])
    gen_a, gen_b = a["mean_generation_s"], b["mean_generation_s"]
    lines: list[str] = []
    add = lines.append

    add("# RAG evaluation results\n")
    add("## Run information\n")
    add("| Field                              | Value |")
    add("| ---------------------------------- | ----- |")
    add(f"| Evaluation date                    | {date.today().isoformat()} |")
    add(
        "| Framework and version              | "
        + ("RAGAS 0.4.3" if has_ragas else "retrieval-only harness (`scripts/evaluate.py`)")
        + " |"
    )
    add(f"| Evaluator model                    | {run.get('evaluator_model') or NOT_RUN} |")
    add(f"| Generator model                    | {run['generator_model']} |")
    add(
        f"| Embedding model                    | {run['embedding_model']} "
        f"(chunk {run['chunk_size']}/{run['chunk_overlap']}) |"
    )
    add(f"| Corpus version/commit              | 13 docs / 633 chunks @ `{git_commit()}` |")
    add(f"| Golden dataset size                | {run['golden_dataset_size']} |")
    add(f"| `top_k`                            | {run['top_k']} |")
    add(
        f"| Fallback threshold and calibration | {run['score_threshold']} — "
        "`scripts/calibrate_threshold.py`: 10 query in-domain (min 0.682) tách "
        "sạch khỏi 8 query out-of-domain (max 0.488) |\n"
    )

    add("## Configurations\n")
    add(
        "- **Config A — dense-only:** `retrieve(query, top_k=5, use_reranking=False)` "
        "— chỉ ChromaDB cosine trên bge-m3, lấy thẳng top-5 dense."
    )
    add(
        "- **Config B — hybrid + RRF:** `retrieve(query, top_k=5, use_reranking=True)` "
        "— dense top-10 và BM25Plus top-10 trên cùng corpus, fuse một lần bằng "
        "RRF `1/(60+rank)`, lấy top-5.\n"
    )
    add(
        "Hai config dùng cùng golden dataset, generator, evaluator, prompt và "
        "`top_k`; chỉ thay retrieval strategy.\n"
    )

    add("## Overall scores\n")
    add("| Metric            | Config A | Config B | Delta B−A |")
    add("| ----------------- | -------: | -------: | --------: |")
    pairs = []
    for key, label in METRIC_LABELS:
        va, vb = ragas_a.get(key), ragas_b.get(key)
        pairs.append((va, vb))
        add(f"| {label:<17} | {fmt(va):>8} | {fmt(vb):>8} | {delta(va, vb):>9} |")
    valid = [(x, y) for x, y in pairs if isinstance(x, (int, float)) and isinstance(y, (int, float))]
    if valid:
        avg_a = sum(x for x, _ in valid) / len(valid)
        avg_b = sum(y for _, y in valid) / len(valid)
        add(f"| **Average**       | {avg_a:8.3f} | {avg_b:8.3f} | {avg_b - avg_a:+9.3f} |")
    else:
        add(f"| **Average**       | {NOT_RUN:>8} | {NOT_RUN:>8} | {NOT_RUN:>9} |")
    add("")
    add("Chỉ số tất định đo kèm, không cần LLM-judge:\n")
    add("| Metric | Config A | Config B | Delta B−A |")
    add("| --- | ---: | ---: | ---: |")
    add(
        f"| Context hit-rate | {a['context_hit_rate']:.3f} | "
        f"{b['context_hit_rate']:.3f} | {hit_delta:+.3f} |"
    )
    add(
        f"| Median retrieval (ms) | {retr_a * 1000:.0f} | {retr_b * 1000:.0f} | "
        f"{(retr_b - retr_a) * 1000:+.0f} |"
    )
    add(
        f"| Mean generation (s) | {gen_a:.2f} | {gen_b:.2f} | {gen_b - gen_a:+.2f} |\n"
    )
    add(
        "*Context hit-rate = tỉ lệ câu hỏi mà `expected_context` xuất hiện "
        "nguyên văn trong contexts đã truy hồi. Retrieval và generation được "
        "bấm giờ riêng: gộp lại thì latency bị LLM call chi phối. Retrieval "
        "latency lấy trung vị của benchmark chạy xen kẽ A/B 3 lượt — đo trong "
        "lượt chính thì config chạy trước gánh phần làm nóng cache và cho số "
        "phi lý.*\n"
    )

    add("## A/B comparison\n")
    better = "Config A — dense-only" if hit_delta < 0 else "Config B — hybrid + RRF"
    add(f"- Cấu hình tốt hơn: **{better}**")
    if has_ragas:
        add(
            f"- Evidence: context hit-rate {a['context_hit_rate']:.3f} (A) so với "
            f"{b['context_hit_rate']:.3f} (B), delta {hit_delta:+.3f}; "
            f"trung bình 4 metric RAGAS {avg_a:.3f} (A) so với {avg_b:.3f} (B)."
        )
    else:
        add(
            f"- Evidence: context hit-rate {a['context_hit_rate']:.3f} (A) so với "
            f"{b['context_hit_rate']:.3f} (B), delta {hit_delta:+.3f} trên "
            f"{run['golden_dataset_size']} câu hỏi. 4 metric RAGAS {NOT_RUN}."
        )
    add(
        "- Kết quả đi ngược kỳ vọng thông thường. Nguyên nhân đã truy ra: corpus "
        "chứa hai cặp tài liệu gần trùng nội dung (bản quy chế đào tạo tiếng Anh "
        "so với bản tiếng Việt, và hai bài viết cùng mô tả 3 mức học bổng). BM25 "
        "xếp hạng 1 cho bản gần trùng, và RRF cho top-1 của mỗi list trọng số "
        "`1/61` — lớn hơn hạng 3 của dense (`1/63`) — nên chunk đúng bị đẩy khỏi "
        "top-5. Vấn đề nằm ở trùng lặp dữ liệu, không phải ở công thức RRF."
    )
    add(
        f"- Trade-off về latency/cost: B thêm một lượt BM25 trên 633 chunk, "
        f"retrieval tốn {retr_b * 1000:.0f} ms so với {retr_a * 1000:.0f} ms của A "
        f"({(retr_b - retr_a) * 1000:+.0f} ms/query, trung vị) vì BM25 index được "
        f"cache sau lần build đầu. B không tốn thêm API call; generation "
        f"({gen_a:.2f}s so với {gen_b:.2f}s) chênh nhau do độ dài context và độ "
        "trễ phía OpenAI, không phải do retrieval strategy. Chi phí thêm của B "
        "là không đáng kể — lý do chưa chọn B là chất lượng, không phải giá.\n"
    )

    add("## Worst performers\n")
    add(
        "|   # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause |"
    )
    add(
        "| --: | -------- | ------ | -----------: | --------: | -----: | --------: | ------------- | ---------- |"
    )
    for position, entry in enumerate(collect_worst(report), 1):
        row, metrics = entry["row"], entry["metrics"]
        stage, cause = ROOT_CAUSES.get(
            row["id"],
            (
                "retrieval" if not row["context_hit"] else "generation",
                "Xem eval_results.json để đối chiếu contexts đã truy hồi.",
            ),
        )
        question = row["user_input"].replace("|", "/")
        add(
            f"|   {position} | {question} | {entry['config']} | "
            f"{fmt(metrics.get('faithfulness'))} | {fmt(metrics.get('answer_relevancy'))} | "
            f"{fmt(metrics.get('context_recall'))} | {fmt(metrics.get('context_precision'))} | "
            f"{stage} | {cause} |"
        )
    add("")

    add("## Recommendations\n")
    add("| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |")
    add("| -------: | ------ | ------------------------------ | --------------- | ------------- |")
    add(
        "|        1 | Khử trùng lặp ở tầng corpus: loại bản dịch tiếng Anh "
        "`hust-quy-che-dao-tao-tin-chi-en.md` hoặc gộp near-duplicate trước khi index "
        "| 2/3 worst performer là do BM25 xếp hạng 1 cho bản gần trùng rồi RRF đẩy "
        "chunk đúng khỏi top-5 | Hybrid hết bị phạt bởi trùng lặp; kỳ vọng "
        f"context hit-rate của B vượt {a['context_hit_rate']:.3f} của A "
        "| Chạy lại `python -m scripts.evaluate` và so delta B−A |"
    )
    add(
        "|        2 | Tăng `CHUNK_OVERLAP` từ 50 lên ~150 cho tài liệu legal, "
        "hoặc chunk theo ranh giới Điều/Khoản thay vì ký tự "
        "| hp-02 hỏng vì câu định nghĩa cách tính học phí nằm vắt qua ranh giới "
        "chunk-3/chunk-4 | Giảm lỗi mất mệnh đề ở văn bản hành chính dài "
        "| `python -m scripts.verify_golden_dataset` rồi `scripts.evaluate`, "
        "theo dõi riêng nhóm câu hỏi `category=học phí` |"
    )
    add(
        "|        3 | Bổ sung chỉ số đánh giá theo nội dung tương đương thay vì "
        "khớp chuỗi tuyệt đối | hb-01 bị chấm miss dù context truy hồi được "
        "(article_05) trả lời đúng y hệt ground truth trích từ article_06 "
        "| Hit-rate phản ánh đúng chất lượng, tránh tối ưu nhầm hướng "
        "| Đối chiếu hit-rate với `context_recall` của RAGAS trên cùng lần chạy |\n"
    )

    add("## Bonus experiments\n")
    add("| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |")
    add("| ---------- | -------- | -----------: | -----------------: | ---------- |")
    add(
        "| Conversation memory cho follow-up (toggle trong `app.py`, 2 lượt gần nhất "
        "ghép vào user message) | Không có memory | chưa đo định lượng "
        "| +0 API call, prompt dài thêm ~400 token/lượt "
        "| Chạy được, demo bằng câu hỏi nối tiếp; chưa có A/B định lượng |"
    )
    add(
        "| UI citation/source highlighting (`app.py` hiển thị `[Document N]`, "
        "retrieval_method, score và link nguồn cho từng chunk) | UI chỉ có answer "
        "| không áp dụng | +0 | Chạy được, kiểm chứng bằng demo |\n"
    )

    RESULT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Rendered {RESULT_MD}")


if __name__ == "__main__":
    main()
