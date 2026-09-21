# Individual contribution report

## Thông tin

- Họ và tên: Bùi Đức Thông
- Mã học viên: 2A202602931
- Nhóm: <điền tên/số nhóm>
- Repository/branch: `K4-L3A-RAG-Pipeline-BuiDucVinh-2A202602801` / `main`

## Phần việc đã thực hiện

> Chỉ kê khai việc đối chiếu được bằng file, commit, PR, test hoặc kết quả
> evaluation. Xoá hết các dòng không phải phần của mình — bảng này được chấm
> bằng cách mở đúng file/commit ra xem.

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| Kiểm chứng trùng lặp corpus | Gỡ bản dịch EN, reindex, chạy lại A/B 5 config, so với RESULT.md cũ | `data/standardized/legal/`, `group_project/evaluation/RESULT.md` | Chưa bắt đầu |
|  |  |  | Done / Partial / Blocked |
|  |  |  | Done / Partial / Blocked |

<details>
<summary>Các deliverable hiện có trong repo (để tra đường dẫn khi điền bảng)</summary>

| Deliverable | File |
|---|---|
| Thu thập văn bản pháp quy (PDF) | `src/task1_collect_legal_docs.py` |
| Crawl tin tức hust.edu.vn / ctt.hust.edu.vn | `src/task2_crawl_news.py` |
| Chuẩn hoá sang Markdown + front matter | `src/task3_convert_markdown.py` |
| Chunking 500/50 + index bge-m3 vào ChromaDB | `src/task4_chunking_indexing.py` |
| Embedding / dense search | `src/task5`, `src/task6` |
| BM25 + RRF | `src/task7_hybrid_search.py` |
| PageIndex fallback | `src/task8_pageindex.py` |
| Retrieval pipeline | `src/task9_retrieval_pipeline.py` |
| Generation + citation `[Document N]` | `src/task10_generation.py` |
| Chatbot Streamlit | `app.py` |
| Golden dataset 18 câu + verify | `group_project/evaluation/`, `scripts/verify_golden_dataset.py` |
| A/B 5 config + RAGAS | `scripts/evaluate.py`, `scripts/render_result.py` |
| Bonus — cross-encoder rerank | `src/bonus_cross_encoder.py` |
| Bonus — HyDE query expansion | `src/bonus_query_expansion.py` |
| Bonus — conversation memory + A/B | `src/bonus_conversation_memory.py`, `scripts/evaluate_memory.py` |
| Test | `tests/test_contracts.py`, `tests/test_acceptance.py`, `tests/test_bonus.py` |

Lấy commit tương ứng: `git log --oneline -- <đường dẫn file>`

</details>

## Quyết định kỹ thuật quan trọng

Tối đa hai quyết định mà tôi **trực tiếp** tham gia. Mỗi mục cần một con số
hoặc một lỗi cụ thể làm bằng chứng, không viết lý thuyết chung.

1. **Quyết định:**
   **Lý do/evidence:**
   **Trade-off:**

2. **Quyết định:**
   **Lý do/evidence:**
   **Trade-off:**

## Kiểm thử và kết quả

- Test hoặc query tôi đã dùng:
- Kết quả trước/sau nếu có:
- Lỗi đã phát hiện và cách xử lý:

## Điều còn hạn chế

- Một hạn chế cụ thể của phần tôi làm:
- Nếu có thêm thời gian, thay đổi đầu tiên tôi sẽ thực hiện:

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải
thích hoặc chạy lại trong buổi demo.

- Ngày:
- Tên thành viên: Bùi Đức Thông
