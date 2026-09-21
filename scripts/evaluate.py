"""Đánh giá RAG pipeline và so sánh A/B dense-only với hybrid + RRF.

Hai config chỉ khác nhau ở retrieval strategy; golden dataset, generator,
evaluator, prompt và top_k giữ nguyên.

    Config A — dense-only:     retrieve(..., use_reranking=False)
    Config B — hybrid + RRF:   retrieve(..., use_reranking=True)

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

from src.task9_retrieval_pipeline import SCORE_THRESHOLD, retrieve
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

CONFIGS = {
    "A_dense_only": {"use_reranking": False, "label": "Config A — dense-only"},
    "B_hybrid_rrf": {"use_reranking": True, "label": "Config B — hybrid + RRF"},
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


def run_config(cases: list[dict], use_reranking: bool, generate: bool) -> list[dict]:
    """Chạy pipeline cho toàn bộ golden dataset với một retrieval config."""
    rows: list[dict] = []
    for index, case in enumerate(cases, 1):
        # Đo riêng retrieval và generation: latency end-to-end bị chi phối bởi
        # LLM call nên gộp lại thì không so sánh được hai retrieval strategy.
        started = time.time()
        chunks = retrieve(
            case["question"], top_k=TOP_K, use_reranking=use_reranking
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
            }
        )
        print(f"  [{index:>2}/{len(cases)}] {case['id']} hit={rows[-1]['context_hit']}")
    return rows


def benchmark_retrieval(cases: list[dict], repeats: int = 3) -> dict[str, float]:
    """Đo latency retrieval của hai config, xen kẽ và lấy trung vị.

    Đo trong lượt chạy chính thì config chạy trước gánh phần làm nóng cache
    của Chroma và của OS, cho ra kết quả phi lý (dense-only chậm hơn hybrid dù
    làm ít việc hơn). Xen kẽ A/B trên từng query và lấy trung vị loại bỏ
    phần lớn bias đó.
    """
    import statistics

    samples: dict[str, list[float]] = {name: [] for name in CONFIGS}
    for _ in range(repeats):
        for case in cases:
            for name, config in CONFIGS.items():
                started = time.time()
                retrieve(
                    case["question"],
                    top_k=TOP_K,
                    use_reranking=config["use_reranking"],
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Chỉ đo retrieval hit-rate, không gọi LLM generation/judge",
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

    cases = json.loads(GOLDEN_DATASET.read_text(encoding="utf-8"))
    print(f"Golden dataset: {len(cases)} câu hỏi")

    # Nạp sẵn embedding model và BM25 index: nếu không warm-up, config chạy
    # trước sẽ gánh toàn bộ chi phí khởi tạo và phép đo latency vô nghĩa.
    print("Warm-up retrieval...\n")
    retrieve("warm up", top_k=TOP_K)

    report = {
        "run": {
            "generator_model": f"{LLM_PROVIDER}/{LLM_MODEL}",
            "evaluator_model": EVALUATOR_MODEL if not args.retrieval_only else None,
            "embedding_model": EMBEDDING_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "top_k": TOP_K,
            "score_threshold": SCORE_THRESHOLD,
            "golden_dataset_size": len(cases),
            "retrieval_only": args.retrieval_only,
        },
        "configs": {},
    }

    for name, config in CONFIGS.items():
        print(f"=== {config['label']} ===")
        rows = run_config(cases, config["use_reranking"], not args.retrieval_only)
        summary = {
            "label": config["label"],
            "context_hit_rate": sum(row["context_hit"] for row in rows) / len(rows),
            "mean_retrieval_s": sum(row["retrieval_s"] for row in rows) / len(rows),
            "mean_generation_s": sum(row["generation_s"] for row in rows) / len(rows),
            "rows": rows,
        }
        if not args.retrieval_only:
            print("  chấm điểm bằng RAGAS...")
            summary["ragas"] = score_with_ragas(rows)
        report["configs"][name] = summary
        print(f"  context hit-rate = {summary['context_hit_rate']:.2%}\n")

    print("Benchmark retrieval latency (xen kẽ A/B, 3 lượt)...")
    medians = benchmark_retrieval(cases)
    for name, value in medians.items():
        report["configs"][name]["median_retrieval_s"] = value
        print(f"  {name}: {value * 1000:.0f} ms (median)")
    print()

    RESULTS_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved: {RESULTS_JSON}")

    a, b = report["configs"]["A_dense_only"], report["configs"]["B_hybrid_rrf"]
    print("\n| Metric | Config A | Config B | Delta B−A |")
    print("| --- | ---: | ---: | ---: |")
    if not args.retrieval_only:
        for key in METRIC_KEYS:
            va, vb = a["ragas"].get(key), b["ragas"].get(key)
            if va is None or vb is None:
                continue
            print(f"| {key} | {va:.3f} | {vb:.3f} | {vb - va:+.3f} |")
    print(
        f"| context hit-rate | {a['context_hit_rate']:.3f} | "
        f"{b['context_hit_rate']:.3f} | "
        f"{b['context_hit_rate'] - a['context_hit_rate']:+.3f} |"
    )
    print(
        f"| median retrieval (ms) | {a['median_retrieval_s'] * 1000:.0f} | "
        f"{b['median_retrieval_s'] * 1000:.0f} | "
        f"{(b['median_retrieval_s'] - a['median_retrieval_s']) * 1000:+.0f} |"
    )
    print(
        f"| mean generation (s) | {a['mean_generation_s']:.2f} | "
        f"{b['mean_generation_s']:.2f} | "
        f"{b['mean_generation_s'] - a['mean_generation_s']:+.2f} |"
    )


if __name__ == "__main__":
    main()
