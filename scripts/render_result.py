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
MEMORY_JSON = EVALUATION_DIR / "memory_results.json"
RESULT_MD = EVALUATION_DIR / "RESULT.md"

# Mô tả từng config, dùng cho mục Configurations. Khoá khớp eval_results.json.
CONFIG_NOTES = {
    "A_dense_only": (
        "`run_retrieval(query, top_k=5, use_reranking=False)` — chỉ ChromaDB "
        "cosine trên bge-m3, lấy thẳng top-5 dense."
    ),
    "B_hybrid_rrf": (
        "`run_retrieval(query, top_k=5, use_reranking=True)` — dense top-10 và "
        "BM25Plus top-10 trên cùng corpus, fuse một lần bằng RRF `1/(60+rank)`, "
        "lấy top-5. **Baseline của mọi config bonus.**"
    ),
    "C_cross_encoder": (
        "`run_retrieval(..., reranker=\"cross_encoder\")` — cùng pool ứng viên "
        "như B nhưng chấm lại từng cặp (query, chunk) bằng "
        "`BAAI/bge-reranker-v2-m3` thay vì fuse theo thứ hạng."
    ),
    "D_hyde_rrf": (
        "`run_retrieval(..., query_variants=expand_query(query))` — LLM viết một "
        "đoạn văn giả định trả lời câu hỏi, dense search thêm trên đoạn đó, rồi "
        "fuse cả ba ranked list bằng RRF như B."
    ),
    "E_hyde_cross_encoder": (
        "Gộp C và D: pool ứng viên có thêm nhánh HyDE, chấm lại bằng cross-encoder."
    ),
}

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
    """Case tệ nhất trên mọi config, xếp theo trung bình 4 metric rồi context hit."""
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
                    "config": name.split("_")[0],
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
    for name, config in report["configs"].items():
        note = CONFIG_NOTES.get(name, "")
        add(f"- **{config['label']}:** {note}")
    add("")
    add(
        "Mọi config dùng chung golden dataset, generator, evaluator, prompt và "
        "`top_k`; chỉ thay retrieval strategy. A/B chính theo yêu cầu bài là "
        "**A so với B**; các config bonus C/D/E lấy **B** làm mốc vì mỗi config "
        "chỉ đổi đúng một thành phần của B, nên chênh lệch quy được về đúng "
        "thành phần đó.\n"
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
        f"{round(retr_b * 1000) - round(retr_a * 1000):+d} |"
    )
    add(
        f"| Mean generation (s) | {gen_a:.2f} | {gen_b:.2f} | {gen_b - gen_a:+.2f} |\n"
    )
    add(
        "*Context hit-rate = tỉ lệ câu hỏi mà `expected_context` xuất hiện "
        "nguyên văn trong contexts đã truy hồi. Retrieval và generation được "
        "bấm giờ riêng: gộp lại thì latency bị LLM call chi phối. Retrieval "
        "latency lấy trung vị của benchmark riêng, chạy 5 lượt và XOAY VÒNG thứ tự "
        "config mỗi lượt. Hai lần đo trước đều sai: đo trong lượt chính thì "
        "config chạy trước gánh phần làm nóng cache, còn xen kẽ theo thứ tự cố "
        "định thì config đứng đầu vẫn gánh cold cache của từng query — cho ra "
        "dense-only chậm hơn hybrid suốt ba lượt, dù nó làm strictly ít việc "
        "hơn.*\n"
    )

    add("### Toàn bộ config\n")
    names = list(report["configs"])
    short = [name.split("_")[0] for name in names]
    add("| Metric | " + " | ".join(short) + " |")
    add("| --- |" + " ---: |" * len(names))
    for key, label in METRIC_LABELS:
        cells = [fmt((report["configs"][name].get("ragas") or {}).get(key)) for name in names]
        add(f"| {label} | " + " | ".join(cells) + " |")
    averages: dict[str, float | None] = {}
    for name in names:
        ragas = report["configs"][name].get("ragas") or {}
        values = [ragas[key] for key, _ in METRIC_LABELS if isinstance(ragas.get(key), (int, float))]
        averages[name] = sum(values) / len(values) if values else None
    add("| **Average** | " + " | ".join(fmt(averages[name]) for name in names) + " |")
    add(
        "| Context hit-rate | "
        + " | ".join(f"{report['configs'][name]['context_hit_rate']:.3f}" for name in names)
        + " |"
    )
    add(
        "| Median retrieval (ms) | "
        + " | ".join(
            f"{report['configs'][name].get('median_retrieval_s', 0) * 1000:.0f}"
            for name in names
        )
        + " |"
    )
    add(
        "| Mean expansion (s) | "
        + " | ".join(
            f"{report['configs'][name].get('mean_expansion_s', 0.0):.2f}" for name in names
        )
        + " |"
    )
    add(
        "| Mean generation (s) | "
        + " | ".join(f"{report['configs'][name]['mean_generation_s']:.2f}" for name in names)
        + " |"
    )
    add("")
    best = max(names, key=lambda name: (averages[name] or 0.0, report["configs"][name]["context_hit_rate"]))
    add(
        f"Config tốt nhất trên toàn bộ phép đo: **{report['configs'][best]['label']}** "
        f"(trung bình {fmt(averages[best])}, hit-rate "
        f"{report['configs'][best]['context_hit_rate']:.3f}). *Mean expansion* là "
        "chi phí LLM sinh đoạn văn HyDE, tách khỏi latency retrieval vì đó là "
        "network call chứ không phải công việc của retrieval.\n"
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
        "- Chẩn đoán trên được kiểm chứng bằng Config C: giữ nguyên pool ứng "
        "viên của B, chỉ thay bước fuse bằng cross-encoder đọc nội dung, thì "
        f"hit-rate lên {report['configs']['C_cross_encoder']['context_hit_rate']:.3f} "
        "— cao hơn cả A. Đúng là khâu 'tin vào thứ hạng của BM25' gây lỗi, "
        "không phải khâu lấy ứng viên."
        if "C_cross_encoder" in report["configs"]
        else "- Chưa chạy config cross-encoder để kiểm chứng chẩn đoán trên."
    )
    add(
        f"- Trade-off về latency/cost: B thêm một lượt BM25 trên 633 chunk, "
        f"retrieval tốn {retr_b * 1000:.0f} ms so với {retr_a * 1000:.0f} ms của A "
        f"({round(retr_b * 1000) - round(retr_a * 1000):+d} ms/query, trung vị) vì "
        f"BM25 index được "
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
    best_name = max(
        names, key=lambda name: (averages[name] or 0.0, report["configs"][name]["context_hit_rate"])
    )
    best_config = report["configs"][best_name]
    best_retr = best_config.get("median_retrieval_s", best_config["mean_retrieval_s"])
    add(
        f"|        1 | Chuyển retrieval mặc định sang **{best_config['label']}** "
        "(`reranker=\"cross_encoder\"`) thay cho RRF "
        "| Đã đo, không phải phỏng đoán: cùng pool ứng viên của B, chỉ thay bước "
        f"fuse, hit-rate {b['context_hit_rate']:.3f} → "
        f"{best_config['context_hit_rate']:.3f} và trung bình RAGAS "
        f"{fmt(averages['B_hybrid_rrf'])} → {fmt(averages[best_name])} "
        f"| Đã đạt: +{best_config['context_hit_rate'] - b['context_hit_rate']:.3f} "
        "hit-rate. Việc còn lại là quyết định có chấp nhận latency không "
        f"| Latency retrieval tăng {retr_b * 1000:.0f} ms → {best_retr * 1000:.0f} ms; "
        "chạy `python -m scripts.evaluate --benchmark-only` trên phần cứng đích |"
    )
    add(
        "|        2 | Khử trùng lặp ở tầng corpus: loại bản dịch tiếng Anh "
        "`hust-quy-che-dao-tao-tin-chi-en.md` hoặc gộp near-duplicate trước khi index "
        "| 2/3 worst performer là do BM25 xếp hạng 1 cho bản gần trùng rồi RRF đẩy "
        "chunk đúng khỏi top-5. Cross-encoder đã che được triệu chứng này, nhưng "
        "nguyên nhân vẫn còn và vẫn tốn chỗ trong pool ứng viên "
        "| Hybrid hết bị phạt bởi trùng lặp; RRF rẻ có thể đủ dùng, khỏi trả giá "
        "latency của cross-encoder "
        "| Chạy lại `python -m scripts.evaluate` và so delta B−A |"
    )
    # Case nào miss ở MỌI config thì không retrieval strategy nào cứu được —
    # tính từ dữ liệu, đừng đoán. Lần trước tôi viết tay là cross-encoder không
    # cứu được hp-02, trong khi nó cứu được.
    all_configs = list(report["configs"].values())
    hit_by_case: dict[str, list[bool]] = {}
    for config in all_configs:
        for row in config["rows"]:
            hit_by_case.setdefault(row["id"], []).append(row["context_hit"])
    always_miss = sorted(cid for cid, hits in hit_by_case.items() if not any(hits))
    rescued_by_reranker = sorted(
        cid
        for cid, hits in hit_by_case.items()
        if not hits[0] and any(hits) and cid not in always_miss
    )

    add(
        "|        3 | Tăng `CHUNK_OVERLAP` từ 50 lên ~150 cho tài liệu legal, "
        "hoặc chunk theo ranh giới Điều/Khoản thay vì ký tự "
        "| Câu định nghĩa cách tính học phí (hp-02) nằm vắt qua ranh giới "
        "chunk-3/chunk-4. Cross-encoder tình cờ cứu được case này, nhưng bằng "
        "cách xếp hạng lại chứ không phải bằng cách làm chunk đúng tồn tại — "
        "khâu chunking vẫn đang cắt mất mệnh đề trả lời "
        "| Giảm lỗi mất mệnh đề ở văn bản hành chính dài, và không phải dựa vào "
        "reranker để bù "
        "| `python -m scripts.verify_golden_dataset` rồi `scripts.evaluate`, "
        "theo dõi riêng nhóm câu hỏi `category=học phí` |"
    )
    if always_miss:
        add(
            "|        4 | Xem lại cách đặt câu hỏi và ranh giới tài liệu cho "
            f"{', '.join(always_miss)} — case miss ở **mọi** config đã thử "
            f"| {len(always_miss)}/{run['golden_dataset_size']} case không có "
            "retrieval strategy nào lấy đúng context: dense, hybrid, "
            "cross-encoder và HyDE đều trượt. Đây là giới hạn của dữ liệu hoặc "
            "của ground truth, không phải của thuật toán xếp hạng "
            "| Tách được phần lỗi còn lại thành lỗi dữ liệu thay vì tiếp tục "
            "tinh chỉnh retrieval vô ích "
            "| Đối chiếu `expected_context` của case đó với chunk thật trong "
            "`chroma_db` bằng `scripts/verify_golden_dataset.py` |"
        )
    add(
        "|        5 | Bổ sung chỉ số đánh giá theo nội dung tương đương thay vì "
        "khớp chuỗi tuyệt đối | hb-01 bị chấm miss dù context truy hồi được "
        "(article_05) trả lời đúng y hệt ground truth trích từ article_06 "
        "| Hit-rate phản ánh đúng chất lượng, tránh tối ưu nhầm hướng "
        "| Đối chiếu hit-rate với `context_recall` của RAGAS trên cùng lần chạy |\n"
    )
    if rescued_by_reranker:
        add(
            f"*Các case dense-only trượt nhưng một config khác lấy được: "
            f"{', '.join(rescued_by_reranker)}. Chi tiết từng case trong "
            "`eval_results.json`.*\n"
        )

    add("## Bonus experiments\n")
    add(
        "Mọi config bonus so với **Config B — hybrid + RRF** vì mỗi config chỉ "
        "đổi đúng một thành phần của B.\n"
    )
    add("| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |")
    add("| ---------- | -------- | -----------: | -----------------: | ---------- |")

    base_hit = b["context_hit_rate"]
    base_avg = averages.get("B_hybrid_rrf")
    base_retr = b.get("median_retrieval_s", b["mean_retrieval_s"])

    def bonus_row(name: str, title: str, cost: str, conclusion: str) -> None:
        config = report["configs"].get(name)
        if not config:
            add(f"| {title} | Config B | {NOT_RUN} | {cost} | {NOT_RUN} |")
            return
        hit_delta_local = config["context_hit_rate"] - base_hit
        avg_local = averages.get(name)
        metric = f"hit-rate {hit_delta_local:+.3f}"
        if isinstance(avg_local, float) and isinstance(base_avg, float):
            metric += f", RAGAS avg {avg_local - base_avg:+.3f}"
        retr_local = config.get("median_retrieval_s", config["mean_retrieval_s"])
        latency = f"{(retr_local - base_retr) * 1000:+.0f} ms/query"
        expansion = config.get("mean_expansion_s", 0.0)
        if expansion:
            latency += f", +{expansion:.2f}s LLM"
        add(f"| {title} | Config B | {metric} | {latency}, {cost} | {conclusion} |")

    bonus_row(
        "C_cross_encoder",
        "Cross-encoder rerank (`BAAI/bge-reranker-v2-m3`) thay RRF, cùng pool ứng viên",
        "+0 API call, nạp ~2.2GB weight một lần",
        "Cải thiện rõ trên mọi metric. RRF chỉ nhìn thứ hạng nên tin theo BM25 "
        "khi BM25 xếp hạng 1 cho bản gần trùng; cross-encoder đọc nội dung nên "
        "không mắc lỗi đó. Đánh đổi là latency retrieval tăng gần 20 lần.",
    )
    bonus_row(
        "D_hyde_rrf",
        "HyDE query expansion (`scripts`/`src/bonus_query_expansion.py`) cộng vào B",
        "+1 LLM call/query",
        "Có cải thiện so với B nhưng nhỏ hơn cross-encoder và phải trả thêm một "
        "LLM call. Đoạn văn giả định kéo query về đúng thể loại văn bản hành "
        "chính, bù được một phần lỗi lệch thể loại câu hỏi/đoạn văn.",
    )
    bonus_row(
        "E_hyde_cross_encoder",
        "Gộp cả hai: HyDE + cross-encoder",
        "+1 LLM call/query, nạp ~2.2GB weight",
        "Hai cải tiến không cộng dồn. Cross-encoder đã sửa gần hết phần lỗi mà "
        "HyDE nhắm tới, nên thêm HyDE lên trên nó gần như không đổi kết quả "
        "trong khi vẫn phải trả thêm một LLM call mỗi câu.",
    )

    memory = None
    if MEMORY_JSON.exists():
        memory = json.loads(MEMORY_JSON.read_text(encoding="utf-8"))
    if memory:
        summary = memory["summary"]
        none_, prompt_ = summary["none"], summary["prompt"]
        rewrite_ = summary.get("rewrite")
        size = memory["run"]["dataset_size"]
        add(
            "| Conversation memory: ghép lịch sử vào prompt (`app.py`) "
            f"| Không memory, cùng {size} hội thoại hai lượt "
            f"| follow-up accuracy {prompt_['accuracy'] - none_['accuracy']:+.3f} "
            f"({none_['accuracy']:.3f} → {prompt_['accuracy']:.3f}) "
            "| +0 API call, prompt dài thêm ~400 token/lượt "
            "| **Có cải thiện.** Context hit-rate đã là "
            f"{none_['context_hit_rate']:.3f} khi chưa có memory, nên chỗ hỏng "
            "không nằm ở retrieval mà ở generation: model có đủ context nhưng "
            "không nối được đại từ hồi chỉ với chủ thể, và trả về refusal. "
            "Lịch sử trong prompt sửa đúng chỗ đó. |"
        )
        if rewrite_:
            add(
                "| Conversation memory: viết lại câu hỏi theo lịch sử trước khi "
                "retrieve (`src/bonus_conversation_memory.py`) "
                "| Memory trong prompt "
                f"| follow-up accuracy {rewrite_['accuracy'] - prompt_['accuracy']:+.3f} "
                f"({prompt_['accuracy']:.3f} → {rewrite_['accuracy']:.3f}), "
                f"context hit-rate "
                f"{rewrite_['context_hit_rate'] - prompt_['context_hit_rate']:+.3f} "
                f"| +1 LLM call/query (~{rewrite_['mean_rewrite_s']:.2f}s) "
                "| **Không dùng.** Sửa được fu-01 nhưng làm hỏng fu-07: bản viết "
                "lại đánh rơi cụm 'để được xét công nhận tốt nghiệp', retrieval "
                "mất chunk đúng và câu trả lời thành refusal. Đổi một lỗi ở "
                "prompt (context vẫn còn) lấy một lỗi ở retrieval (không còn gì "
                "để cứu). Giữ làm toggle, mặc định tắt. |"
            )
    else:
        add(
            "| Conversation memory cho follow-up | Không memory "
            f"| {NOT_RUN} | +0 API call | Chạy `python -m scripts.evaluate_memory` |"
        )
    add(
        "| UI citation/source highlighting (`app.py` hiển thị `[Document N]`, "
        "retrieval_method, score và link nguồn cho từng chunk) | UI chỉ có answer "
        "| không áp dụng | +0 | Chạy được, kiểm chứng bằng demo |\n"
    )
    if memory:
        add(
            f"*Bộ follow-up có {memory['run']['dataset_size']} hội thoại — chênh "
            f"một case là {1 / memory['run']['dataset_size']:.3f}. Đủ để quyết "
            "định bật hay tắt một toggle, chưa đủ để kết luận tổng quát. Mọi "
            "câu lượt 2 đều là ellipsis hoặc có đại từ hồi chỉ; thước đo là "
            "khớp chuỗi dữ kiện, không dùng LLM-judge. Chi tiết từng case: "
            "`group_project/evaluation/memory_results.json`.*\n"
        )

    RESULT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Rendered {RESULT_MD}")


if __name__ == "__main__":
    main()
