"""Đánh giá RAG pipeline và so sánh A/B các retrieval strategy.

Mọi config chỉ khác nhau ở retrieval; golden dataset, generator, evaluator,
prompt và top_k giữ nguyên.

    Config A — dense-only:            chỉ ChromaDB cosine
    Config B — hybrid + RRF:          dense + BM25, fuse bằng RRF   (baseline)
    Config C — hybrid + cross-encoder: cùng pool ứng viên như B, chấm lại
                                      bằng BAAI/bge-reranker-v2-m3
    Config D — HyDE + hybrid + RRF:   thêm ranked list dense của đoạn văn
                                      giả định do LLM sinh, rồi fuse như B
    Config E — HyDE + cross-encoder:  gộp cả hai cải tiến

C và D đều lấy B làm mốc so sánh vì chúng chỉ thay đúng một thành phần của
B — so với A thì không tách được phần cải thiện nào do đâu. E trả lời câu
hỏi tiếp theo: hai cải tiến có cộng dồn được không, hay cái sau nuốt mất
phần đóng góp của cái trước.

Metrics:
    - faithfulness, answer relevance, context recall, context precision (RAGAS)
    - context hit-rate: tỉ lệ câu hỏi mà expected_context xuất hiện trong
      contexts đã truy hồi. Đây là thước đo tất định, không cần LLM, dùng để
      đối chiếu khi LLM-judge dao động.

Cách chạy:
    python -m scripts.evaluate                  # đầy đủ, cần OPENAI_API_KEY
    python -m scripts.evaluate --retrieval-only # chỉ đo retrieval, không gọi LLM
"""

import argparse
import json
import os
import re
import time
import unicodedata
from pathlib import Path

from dotenv import load_dotenv

from src.task9_retrieval_pipeline import SCORE_THRESHOLD, run_retrieval
from src.bonus_query_expansion import expand_query
from src.bonus_cross_encoder import RERANKER_MODEL
from src.task10_generation import (
    LLM_MODEL,
    LLM_PROVIDER,
    SYSTEM_PROMPT,
    build_prompt,
    call_llm,
)
from src.task4_chunking_indexing import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
)


load_dotenv()

EVALUATION_DIR = Path(__file__).parent.parent / "group_project" / "evaluation"
GOLDEN_DATASET = EVALUATION_DIR / "golden_dataset.json"
RESULTS_JSON = EVALUATION_DIR / "eval_results.json"

TOP_K = 5
EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", "gpt-4o-mini")
EVALUATOR_EMBEDDING = os.getenv("EVALUATOR_EMBEDDING", "text-embedding-3-small")

BASELINE = "B_hybrid_rrf"

CONFIGS = {
    "A_dense_only": {
        "label": "Config A — dense-only",
        "use_reranking": False,
        "reranker": "rrf",
        "expand": False,
    },
    "B_hybrid_rrf": {
        "label": "Config B — hybrid + RRF",
        "use_reranking": True,
        "reranker": "rrf",
        "expand": False,
    },
    "C_cross_encoder": {
        "label": "Config C — hybrid + cross-encoder",
        "use_reranking": True,
        "reranker": "cross_encoder",
        "expand": False,
    },
    "D_hyde_rrf": {
        "label": "Config D — HyDE + hybrid + RRF",
        "use_reranking": True,
        "reranker": "rrf",
        "expand": True,
    },
    "E_hyde_cross_encoder": {
        "label": "Config E — HyDE + hybrid + cross-encoder",
        "use_reranking": True,
        "reranker": "cross_encoder",
        "expand": True,
    },
}

METRIC_KEYS = [
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "context_precision",
]

# RAGAS đặt tên cột theo class, không theo tên metric chung; ví dụ
# LLMContextPrecisionWithReference -> "llm_context_precision_with_reference".
# Khớp theo substring để đổi version RAGAS không âm thầm mất một metric.
COLUMN_ALIASES = {
    "faithfulness": ("faithfulness",),
    "answer_relevancy": ("answer_relevancy", "response_relevancy"),
    "context_recall": ("context_recall",),
    "context_precision": ("context_precision",),
}


def match_column(columns: list[str], key: str) -> str | None:
    for candidate in COLUMN_ALIASES[key]:
        for column in columns:
            if column == candidate:
                return column
    for candidate in COLUMN_ALIASES[key]:
        for column in columns:
            if candidate in column:
                return column
    return None


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text.lower())).strip()


def context_hit(expected: str, contexts: list[str]) -> bool:
    needle = normalize(expected)
    return any(needle in normalize(context) for context in contexts)


def build_variants(cases: list[dict]) -> dict[str, list[str]]:
    """Sinh sẵn biến thể HyDE cho từng câu hỏi và bấm giờ riêng phần LLM.

    Sinh trước rồi dùng lại cho cả lượt chấm điểm lẫn lượt benchmark latency:
    nếu để run_retrieval tự gọi LLM thì (a) chi phí HyDE lẫn vào latency
    retrieval, và (b) mỗi lượt benchmark lại trả tiền thêm một lần.
    """
    variants: dict[str, list[str]] = {}
    elapsed: dict[str, float] = {}
    for index, case in enumerate(cases, 1):
        started = time.time()
        variants[case["id"]] = expand_query(case["question"])
        elapsed[case["id"]] = time.time() - started
        generated = len(variants[case["id"]]) > 1
        print(f"  [{index:>2}/{len(cases)}] {case['id']} hyde={generated}")
    build_variants.elapsed = elapsed
    return variants


def run_config(
    cases: list[dict],
    config: dict,
    generate: bool,
    variants: dict[str, list[str]] | None = None,
) -> list[dict]:
    """Chạy pipeline cho toàn bộ golden dataset với một retrieval config."""
    rows: list[dict] = []
    for index, case in enumerate(cases, 1):
        case_variants = variants.get(case["id"]) if (config["expand"] and variants) else None

        # Đo riêng retrieval và generation: latency end-to-end bị chi phối bởi
        # LLM call nên gộp lại thì không so sánh được hai retrieval strategy.
        # Chi phí sinh HyDE cũng nằm ngoài mốc này, ghi riêng ở expansion_s.
        started = time.time()
        chunks = run_retrieval(
            case["question"],
            top_k=TOP_K,
            use_reranking=config["use_reranking"],
            reranker=config["reranker"],
            query_variants=case_variants,
        )
        retrieval_s = time.time() - started
        contexts = [chunk["content"] for chunk in chunks]

        answer = ""
        generation_s = 0.0
        if generate and chunks:
            generation_started = time.time()
            try:
                answer = call_llm(
                    SYSTEM_PROMPT, build_prompt(case["question"], chunks)
                )
            except Exception as error:
                print(f"  [{case['id']}] generation failed: {error}")
            generation_s = time.time() - generation_started

        rows.append(
            {
                "id": case["id"],
                "category": case["category"],
                "user_input": case["question"],
                "retrieved_contexts": contexts,
                "reference_contexts": [case["expected_context"]],
                "response": answer,
                "reference": case["expected_answer"],
                "context_hit": context_hit(case["expected_context"], contexts),
                "retrieval_s": round(retrieval_s, 4),
                "generation_s": round(generation_s, 3),
                "expansion_s": round(
                    getattr(build_variants, "elapsed", {}).get(case["id"], 0.0), 3
                )
                if config["expand"]
                else 0.0,
            }
        )
        print(f"  [{index:>2}/{len(cases)}] {case['id']} hit={rows[-1]['context_hit']}")
    return rows


def benchmark_retrieval(
    cases: list[dict],
    variants: dict[str, list[str]] | None = None,
    repeats: int = 5,
) -> dict[str, float]:
    """Đo latency retrieval của mọi config và lấy trung vị.

    Hai nguồn bias đã gặp và cách xử lý:

    1. Đo trong lượt chạy chính thì config chạy trước gánh phần làm nóng cache
       của Chroma và của OS. Vì vậy đo riêng ở đây.
    2. Xen kẽ config theo một thứ tự CỐ ĐỊNH vẫn còn bias: config đứng đầu
       luôn là config chạy đầu tiên cho mỗi query mới, nên vẫn gánh phần cold
       cache của riêng query đó. Đo kiểu này cho ra A (dense-only) chậm hơn B
       (dense + BM25) suốt ba lượt chạy — vô lý, vì A làm strictly ít việc
       hơn. Nên xoay vòng thứ tự config theo từng lượt: mỗi config lần lượt
       đứng ở mọi vị trí.
    """
    import statistics

    names = list(CONFIGS)
    samples: dict[str, list[float]] = {name: [] for name in names}
    for repeat in range(repeats):
        # Xoay vòng: lượt 0 chạy A,B,C,D,E; lượt 1 chạy B,C,D,E,A; ...
        order = names[repeat % len(names):] + names[: repeat % len(names)]
        for case in cases:
            for name in order:
                config = CONFIGS[name]
                case_variants = (
                    variants.get(case["id"])
                    if (config["expand"] and variants)
                    else None
                )
                started = time.time()
                run_retrieval(
                    case["question"],
                    top_k=TOP_K,
                    use_reranking=config["use_reranking"],
                    reranker=config["reranker"],
                    query_variants=case_variants,
                )
                samples[name].append(time.time() - started)
    return {name: statistics.median(values) for name, values in samples.items()}


def score_with_ragas(rows: list[dict]) -> dict[str, float]:
    """Chấm 4 metric bằng RAGAS với LLM-judge và embedding của OpenAI."""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    samples = [
        SingleTurnSample(
            user_input=row["user_input"],
            retrieved_contexts=row["retrieved_contexts"],
            reference_contexts=row["reference_contexts"],
            response=row["response"],
            reference=row["reference"],
        )
        for row in rows
    ]

    judge = LangchainLLMWrapper(ChatOpenAI(model=EVALUATOR_MODEL, temperature=0))
    embedder = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=EVALUATOR_EMBEDDING)
    )

    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextRecall(),
            LLMContextPrecisionWithReference(),
        ],
        llm=judge,
        embeddings=embedder,
    )

    frame = result.to_pandas()
    columns = list(frame.columns)
    scores: dict[str, float] = {}
    renamed: dict[str, str] = {}
    for key in METRIC_KEYS:
        column = match_column(columns, key)
        if column is None:
            print(f"  cảnh báo: RAGAS không trả cột cho metric '{key}'")
            continue
        renamed[column] = key
        scores[key] = float(frame[column].astype(float).mean(skipna=True))

    per_case = frame[["user_input"] + list(renamed)].rename(columns=renamed)
    scores["_per_case"] = per_case.to_dict(orient="records")
    return scores


def rescore() -> None:
    """Chấm lại 4 metric từ answer đã lưu — dùng khi chỉ sửa phần chấm điểm."""
    report = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    for name, config in report["configs"].items():
        print(f"=== {config['label']} — chấm lại bằng RAGAS ===")
        config["ragas"] = score_with_ragas(config["rows"])
        for key in METRIC_KEYS:
            if key in config["ragas"]:
                print(f"  {key:<18} {config['ragas'][key]:.3f}")
    report["run"]["evaluator_model"] = EVALUATOR_MODEL
    report["run"]["retrieval_only"] = False
    RESULTS_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nSaved: {RESULTS_JSON}")


def benchmark_only() -> None:
    """Đo lại latency trên eval_results.json đã có, giữ nguyên phần chấm điểm."""
    report = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    cases = json.loads(GOLDEN_DATASET.read_text(encoding="utf-8"))

    print("Warm-up retrieval...")
    run_retrieval("warm up", top_k=TOP_K)
    if any(config["reranker"] == "cross_encoder" for config in CONFIGS.values()):
        print("Warm-up cross-encoder...")
        run_retrieval("warm up", top_k=TOP_K, reranker="cross_encoder")

    variants: dict[str, list[str]] = {}
    if any(config["expand"] for config in CONFIGS.values()):
        print("Sinh lại biến thể HyDE cho phép đo latency...")
        variants = build_variants(cases)

    print("\nBenchmark retrieval latency (xoay vòng thứ tự config, 5 lượt)...")
    medians = benchmark_retrieval(cases, variants)
    for name, value in medians.items():
        if name in report["configs"]:
            report["configs"][name]["median_retrieval_s"] = value
        print(f"  {name}: {value * 1000:.0f} ms (median)")

    RESULTS_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nSaved: {RESULTS_JSON}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Chỉ đo retrieval hit-rate, không gọi LLM generation/judge",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Chỉ đo lại latency và cập nhật eval_results.json, không sinh lại answer",
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Chấm lại bằng RAGAS từ eval_results.json, không sinh lại answer",
    )
    args = parser.parse_args()

    if args.rescore:
        rescore()
        return

    if args.benchmark_only:
        benchmark_only()
        return

    cases = json.loads(GOLDEN_DATASET.read_text(encoding="utf-8"))
    print(f"Golden dataset: {len(cases)} câu hỏi")

    # Nạp sẵn embedding model và BM25 index: nếu không warm-up, config chạy
    # trước sẽ gánh toàn bộ chi phí khởi tạo và phép đo latency vô nghĩa.
    print("Warm-up retrieval...")
    run_retrieval("warm up", top_k=TOP_K)
    if any(config["reranker"] == "cross_encoder" for config in CONFIGS.values()):
        # Cross-encoder nạp ~2.2GB weight ở lần predict đầu; không warm-up thì
        # toàn bộ chi phí đó rơi vào query đầu tiên của config C.
        print("Warm-up cross-encoder...")
        run_retrieval("warm up", top_k=TOP_K, reranker="cross_encoder")
    print()

    needs_expansion = any(config["expand"] for config in CONFIGS.values())
    variants: dict[str, list[str]] = {}
    if needs_expansion and not args.retrieval_only:
        print("=== Sinh biến thể HyDE (1 LLM call mỗi câu) ===")
        variants = build_variants(cases)
        hyde_seconds = list(getattr(build_variants, "elapsed", {}).values())
        print(
            f"  trung bình {sum(hyde_seconds) / len(hyde_seconds):.2f}s/câu\n"
            if hyde_seconds
            else ""
        )
    elif needs_expansion:
        print("Bỏ qua HyDE: --retrieval-only không gọi LLM.\n")

    report = {
        "run": {
            "generator_model": f"{LLM_PROVIDER}/{LLM_MODEL}",
            "evaluator_model": EVALUATOR_MODEL if not args.retrieval_only else None,
            "embedding_model": EMBEDDING_MODEL,
            "reranker_model": RERANKER_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "top_k": TOP_K,
            "score_threshold": SCORE_THRESHOLD,
            "golden_dataset_size": len(cases),
            "retrieval_only": args.retrieval_only,
            "baseline": BASELINE,
        },
        "configs": {},
    }

    for name, config in CONFIGS.items():
        if config["expand"] and not variants:
            print(f"=== {config['label']} — bỏ qua (không có biến thể HyDE) ===\n")
            continue
        print(f"=== {config['label']} ===")
        rows = run_config(cases, config, not args.retrieval_only, variants)
        summary = {
            "label": config["label"],
            "context_hit_rate": sum(row["context_hit"] for row in rows) / len(rows),
            "mean_retrieval_s": sum(row["retrieval_s"] for row in rows) / len(rows),
            "mean_generation_s": sum(row["generation_s"] for row in rows) / len(rows),
            "mean_expansion_s": sum(row["expansion_s"] for row in rows) / len(rows),
            "rows": rows,
        }
        if not args.retrieval_only:
            print("  chấm điểm bằng RAGAS...")
            summary["ragas"] = score_with_ragas(rows)
        report["configs"][name] = summary
        print(f"  context hit-rate = {summary['context_hit_rate']:.2%}\n")

    active = list(report["configs"])
    print(f"Benchmark retrieval latency ({len(active)} config, xoay vòng, 5 lượt)...")
    medians = benchmark_retrieval(cases, variants)
    for name in active:
        report["configs"][name]["median_retrieval_s"] = medians[name]
        print(f"  {name}: {medians[name] * 1000:.0f} ms (median)")
    print()

    RESULTS_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved: {RESULTS_JSON}")

    print_summary(report, args.retrieval_only)


def print_summary(report: dict, retrieval_only: bool) -> None:
    """In bảng tổng hợp mọi config kèm delta so với baseline."""
    configs = report["configs"]
    names = list(configs)
    baseline = report["run"].get("baseline", BASELINE)

    def row(title: str, getter, fmt: str = "{:.3f}") -> str:
        cells = []
        for name in names:
            value = getter(configs[name])
            cells.append("n/a" if value is None else fmt.format(value))
        return f"| {title} | " + " | ".join(cells) + " |"

    header = "| Metric | " + " | ".join(name.split("_")[0] for name in names) + " |"
    print("\n" + header)
    print("| --- |" + " ---: |" * len(names))
    if not retrieval_only:
        for key in METRIC_KEYS:
            print(row(key, lambda config: config.get("ragas", {}).get(key)))
    print(row("context hit-rate", lambda config: config["context_hit_rate"]))
    print(
        row(
            "median retrieval (ms)",
            lambda config: config["median_retrieval_s"] * 1000,
            "{:.0f}",
        )
    )
    print(
        row(
            "mean expansion (s)",
            lambda config: config.get("mean_expansion_s", 0.0),
            "{:.2f}",
        )
    )
    print(
        row("mean generation (s)", lambda config: config["mean_generation_s"], "{:.2f}")
    )

    if baseline not in configs:
        return
    print(f"\nDelta so với baseline ({configs[baseline]['label']}):")
    for name in names:
        if name == baseline:
            continue
        delta_hit = (
            configs[name]["context_hit_rate"] - configs[baseline]["context_hit_rate"]
        )
        line = f"  {name:<18} hit-rate {delta_hit:+.3f}"
        if not retrieval_only:
            scores = [
                configs[name]["ragas"].get(key, 0.0)
                for key in METRIC_KEYS
                if key in configs[name].get("ragas", {})
            ]
            base_scores = [
                configs[baseline]["ragas"].get(key, 0.0)
                for key in METRIC_KEYS
                if key in configs[baseline].get("ragas", {})
            ]
            if scores and base_scores:
                delta_avg = sum(scores) / len(scores) - sum(base_scores) / len(
                    base_scores
                )
                line += f"   RAGAS avg {delta_avg:+.3f}"
        print(line)


if __name__ == "__main__":
    main()
