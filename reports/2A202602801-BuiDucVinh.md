# Individual contribution report

## Thông tin

- Họ và tên: Bùi Đức Vinh
- Mã học viên: 2A202602801
- Nhóm: làm cá nhân toàn bộ pipeline
- Repository/branch: `K4-L3A-RAG-Pipeline-BuiDucVinh-2A202602801` / `main`

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| Thu thập dữ liệu | Chọn đề tài dịch vụ sinh viên HUST, tải 4 PDF quy chế/quyết định, crawl 9 bài từ hust.edu.vn và ctt.hust.edu.vn | `src/task1_collect_legal_docs.py`, `src/task2_crawl_news.py` | Done |
| Chuẩn hóa | Convert PDF/JSON sang Markdown kèm front matter `title/source/doc_type/url` | `src/task3_convert_markdown.py` | Done |
| Chunking & index | Recursive 500/50, bge-m3, ChromaDB cosine, ID ổn định để upsert không nhân bản | `src/task4_chunking_indexing.py` | Done |
| Dense + BM25 + RRF | Dense trả cosine score gốc; BM25Plus trên cùng corpus Chroma; RRF `1/(k+rank)`, fuse một lần | `src/task5..task7` | Done |
| Fallback & pipeline | PageIndex vectorless có cache doc ID; fallback so với cosine score gốc; provider lỗi không làm sập pipeline | `src/task8`, `src/task9` | Done (PageIndex chưa chạy thật: không có API key) |
| Generation có citation | Gán nhãn `[Document N]` trước khi reorder, context có title/source, dispatch 3 provider, safe refusal | `src/task10_generation.py` | Done |
| Chatbot | Streamlit hiển thị answer, nguồn, retrieval method, score, link nguồn; toggle hybrid/dense và conversation memory | `app.py` | Done |
| Hiệu chỉnh & đánh giá | Script calibrate threshold, script kiểm chứng ground truth, script A/B 18 câu | `scripts/` | Done |
| Golden dataset & báo cáo | 18 Q&A trích từ corpus, 18/18 được script xác minh là có thật trong chunk đã index | `group_project/evaluation/` | Done |

## Quyết định kỹ thuật quan trọng

1. **Quyết định:** Dùng BM25Plus thay cho BM25Okapi cho lexical search.
   **Lý do/evidence:** idf của Okapi là `log(N-n+0.5) - log(n+0.5)`, bằng 0 khi
   term xuất hiện ở khoảng một nửa số document. Corpus của tôi nhỏ (633 chunk)
   và rất đồng chủ đề nên Okapi triệt tiêu đúng những term đặc trưng của
   domain; test contract với corpus 2 document cho toàn bộ score bằng 0.
   BM25Plus dùng `log((N+1)/n)` nên idf luôn dương.
   **Trade-off:** BM25Plus cộng một hằng số delta cho mọi document kể cả
   document không chứa term nào, nên không lọc "liên quan" bằng `score > 0`
   được nữa; tôi lọc bằng giao token giữa query và chunk.

2. **Quyết định:** Gán nhãn citation `[Document N]` trước bước reorder chống
   lost-in-the-middle, thay vì đánh số theo vị trí trong context.
   **Lý do/evidence:** `sources` bắt buộc sort theo score giảm dần, còn context
   sau reorder có thứ tự `[0,2,4,3,1]`. Nếu đánh số sau reorder thì
   `[Document 2]` trong câu trả lời trỏ vào chunk khác với `sources[1]` và
   citation không còn đối chiếu được.
   **Trade-off:** Số hiệu trong context không còn tăng dần theo thứ tự đọc,
   phải truyền nhãn qua một field phụ `citation_index`.

## Kiểm thử và kết quả

- Test đã dùng: `pytest -q` → 20/20 pass (15 contract + 5 acceptance);
  `python -m scripts.verify_golden_dataset` → 18/18 expected_context tìm thấy
  trong corpus; `python -m scripts.evaluate --retrieval-only` cho A/B.
- Kết quả trước/sau: crawl cả trang cho ~40k ký tự/bài, trong đó ~95% là menu
  điều hướng lặp trên mọi URL. Sau khi khai báo selector nội dung
  (`div.bodytext`, `div.col-md-9`), mỗi bài còn 0.8k–8.6k ký tự nội dung thật.
- Lỗi đã phát hiện và cách xử lý: (1) BM25Okapi trả score 0 trên corpus nhỏ →
  chuyển BM25Plus; (2) ChromaDB từ chối `None` trong metadata → chuẩn hóa
  `url=None` thành chuỗi rỗng khi upsert; (3) title của tài liệu legal lấy từ
  dòng đầu PDF đều là "BỘ GIÁO DỤC VÀ ĐÀO TẠO", citation không phân biệt được
  văn bản → khai báo title tường minh trong `SOURCES`.

## Điều còn hạn chế

- Hạn chế cụ thể: hybrid + RRF hiện **kém hơn** dense-only trên context
  hit-rate (0.778 so với 0.889) và trên trung bình 4 metric RAGAS (0.771 so
  với 0.789; hybrid chỉ thắng ở context precision). Nguyên nhân là BM25 xếp hạng cao bản dịch
  tiếng Anh gần trùng nội dung (`hust-quy-che-dao-tao-tin-chi-en.md`), và RRF
  cho top-1 của BM25 trọng số `1/61` — đủ để đẩy chunk đúng ở hạng 3 của dense
  ra khỏi top-5.
- Nếu có thêm thời gian, thay đổi đầu tiên: khử trùng lặp ở tầng corpus (loại
  bản dịch tiếng Anh hoặc gộp near-duplicate trước khi index), rồi đo lại A/B
  trước khi đụng tới tham số `k` của RRF.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải
thích hoặc chạy lại trong buổi demo.

- Ngày: 2026-09-21
- Tên thành viên: Bùi Đức Vinh
