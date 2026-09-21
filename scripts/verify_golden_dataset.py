"""Kiểm tra mọi expected_context của golden dataset thật sự có trong corpus.

Golden dataset chỉ đáng tin khi ground truth trích từ chính tài liệu đã index;
script này chặn trường hợp câu hỏi được viết theo trí nhớ thay vì theo nguồn.

    python -m scripts.verify_golden_dataset
"""

import json
import re
import unicodedata
from pathlib import Path

from src.task6_lexical_search import load_corpus


DATASET = Path(__file__).parent.parent / "group_project" / "evaluation" / "golden_dataset.json"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    cases = json.loads(DATASET.read_text(encoding="utf-8"))
    corpus = [normalize(item["content"]) for item in load_corpus()]

    missing = []
    for case in cases:
        needle = normalize(case["expected_context"])
        hits = sum(needle in chunk for chunk in corpus)
        status = "OK " if hits else "MISS"
        if not hits:
            missing.append(case["id"])
        print(f"{status} {case['id']:6s} hits={hits:<3d} {case['question'][:58]}")

    print(f"\n{len(cases) - len(missing)}/{len(cases)} expected_context tìm thấy trong corpus")
    if missing:
        raise SystemExit(f"Không grounded: {', '.join(missing)}")


if __name__ == "__main__":
    main()
