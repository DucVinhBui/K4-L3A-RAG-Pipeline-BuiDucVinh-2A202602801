# Day 8 — RAG Pipeline: Dịch vụ sinh viên Đại học Bách khoa Hà Nội

Chatbot RAG trả lời câu hỏi về học phí, học bổng, quy chế đào tạo và dịch vụ
sinh viên của Đại học Bách khoa Hà Nội, dựa trên bộ tài liệu tự thu thập từ
nguồn công khai. Hybrid retrieval (dense + BM25 + RRF), fallback vectorless,
generation có citation đối chiếu được, giao diện chat Streamlit và báo cáo
đánh giá A/B.

## Đề tài và dữ liệu

| Loại | Số lượng | Nguồn |
| --- | ---: | --- |
| Tài liệu chính sách (PDF) | 4 | `ctt.hust.edu.vn`, `hust.edu.vn` — quy chế đào tạo 2023 & 2025, quyết định học phí 2025-2026, credit-based training regulation (EN) |
| Bài viết/thông báo | 9 | `hust.edu.vn`, `ctt.hust.edu.vn` — học phí, học bổng KKHT và Trần Đại Nghĩa, kế hoạch học tập, gửi xe, lễ tốt nghiệp, hỗ trợ sinh viên |

Sau chuẩn hóa: 13 Markdown (~246k ký tự) → **633 chunks** trong ChromaDB.
Mọi file `data/standardized/**.md` mang front matter `title/source/doc_type/url`
nên metadata nguồn đi xuyên suốt tới citation trong câu trả lời.

## Kiến trúc pipeline

```
task1 tải PDF ─┐
               ├─ task3 chuẩn hóa Markdown + front matter
task2 crawl ───┘            │
                            ▼
                 task4 chunk (500/50) → embed (bge-m3) → ChromaDB (cosine)
                            │
              ┌─────────────┴─────────────┐
       task5 dense (cosine)        task6 BM25Plus
              └─────────────┬─────────────┘
                     task7 RRF (k=60, fuse một lần)
                            │
                task9 fallback theo cosine score gốc
                       │ (< 0.59)        │ (≥ 0.59)
                task8 PageIndex      hybrid results
                            │
                task10 reorder → context → LLM → answer + citation
                            │
                         app.py (Streamlit)
```

Quyết định thiết kế đáng chú ý:

- **BM25Plus thay vì BM25Okapi.** idf của Okapi bằng 0 khi term xuất hiện ở
  khoảng một nửa corpus và âm khi phổ biến hơn. Corpus ở đây nhỏ và đồng chủ
  đề nên Okapi triệt tiêu đúng các term đặc trưng ("sinh viên", "học phí").
- **Nhãn citation gán trước khi reorder.** Nếu đánh số sau reorder,
  `[Document 2]` trong câu trả lời sẽ trỏ sai phần tử của `sources` — trong
  khi `sources` bắt buộc sort theo score giảm dần.
- **Fallback so với cosine score gốc, không phải RRF score.** RRF score chỉ
  phản ánh thứ hạng: query ngoài domain vẫn cho top-1 đúng `1/61` như query
  đúng domain, nên không phân biệt được gì.
- **Selector nội dung khi crawl.** Crawl cả trang thì ~95% là menu điều hướng
  lặp trên mọi URL; mỗi nguồn khai báo `selector` trỏ vào khối bài viết.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
python -m playwright install chromium
cp .env.example .env
```

Điền API key cần dùng trong `.env`; không commit file này.
Cấu hình đang dùng: `LLM_PROVIDER=openai`, `LLM_MODEL=gpt-4o-mini`,
`EMBEDDING_PROVIDER=sentence_transformers`, `EMBEDDING_MODEL=BAAI/bge-m3`
(tải ~2.2GB lần đầu), `SCORE_THRESHOLD=0.59`.

```bash
# 1. Thu thập và chuẩn hoá (chạy lại an toàn, không tạo bản trùng)
python -m src.task1_collect_legal_docs
python -m src.task2_crawl_news
python -m src.task3_convert_markdown

# 2. Index và kiểm tra contract
python -m src.task4_chunking_indexing
pytest -q

# 3. Chạy sản phẩm
streamlit run app.py
```

## Hiệu chỉnh ngưỡng fallback

`SCORE_THRESHOLD` không có con số đúng cho mọi corpus — phải đo trên chính
corpus của nhóm:

```bash
python -m scripts.calibrate_threshold
```

Kết quả đo trên 10 query in-domain và 8 query out-of-domain:

| Nhóm query | min | mean | max |
| --- | ---: | ---: | ---: |
| In-domain | 0.6820 | 0.7199 | 0.7604 |
| Out-of-domain | 0.3769 | 0.4236 | 0.4882 |

Hai phân phối tách sạch trong khoảng `(0.4882, 0.6820)` → chọn **0.59**.

## Đánh giá

```bash
python -m scripts.verify_golden_dataset          # mọi ground truth phải có thật trong corpus
python -m scripts.evaluate --retrieval-only      # hit-rate + latency, không tốn API
python -m scripts.evaluate                       # thêm 4 metric RAGAS (cần OPENAI_API_KEY)
```

Golden dataset: 18 câu hỏi trong `group_project/evaluation/golden_dataset.json`,
mọi `expected_context` được script kiểm chứng là trích đúng từ corpus đã index.
Kết quả và phân tích lỗi: [`group_project/evaluation/RESULT.md`](group_project/evaluation/RESULT.md).

## Kiểm tra

```bash
# Contract tests
pytest tests/test_contracts.py -q

# Acceptance tests
pytest tests/test_acceptance.py -q

# Toàn bộ
pytest -q
```

## Cấu trúc

```
src/task1..task10        pipeline theo docs/MODULE_CONTRACTS.md
src/contracts.py         schema + validator dùng chung
scripts/calibrate_threshold.py   hiệu chỉnh ngưỡng fallback
scripts/verify_golden_dataset.py kiểm chứng ground truth
scripts/evaluate.py      A/B dense-only vs hybrid + RRF
app.py                   chatbot Streamlit
data/landing/            file gốc (PDF, JSON crawl)
data/standardized/       Markdown + front matter
```

## Tài liệu

- [Module contracts](docs/MODULE_CONTRACTS.md): schema, interface và invariant mà code/test nên tuân theo.
- [Step-by-step guide](docs/STEP_BY_STEP.md): thứ tự triển khai và tiêu chí hoàn thành từng bước.
- [Grading rubric](docs/GRADING_RUBRIC.md): Rubric thang điểm.
- [Individual report](group_project/ịndividual/INDIVIDUAL_REPORT.md): template báo cáo cá nhân.
- [Suggested topics](docs/SUGGESTED_TOPICS.md): danh sách chủ đề tham khảo, không bắt buộc.
