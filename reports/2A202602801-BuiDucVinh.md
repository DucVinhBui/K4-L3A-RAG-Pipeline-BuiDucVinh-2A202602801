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
| Chatbot | Streamlit hiển thị answer, nguồn, retrieval method, score, link nguồn; toggle hybrid/dense, cross-encoder, HyDE, memory | `app.py` | Done |
| Hiệu chỉnh & đánh giá | Script calibrate threshold, kiểm chứng ground truth, A/B 5 config, A/B memory, sinh RESULT.md | `scripts/` | Done |
| Golden dataset & báo cáo | 18 Q&A trích từ corpus, 18/18 được script xác minh là có thật trong chunk đã index | `group_project/evaluation/` | Done |
| Bonus — reranker nâng cao | Cross-encoder `bge-reranker-v2-m3` chấm lại pool ứng viên, A/B với RRF | `src/bonus_cross_encoder.py` | Done — hit-rate 0.778 → 0.944 |
| Bonus — query expansion | HyDE: LLM viết đoạn giả định, dense search thêm trên đoạn đó, fuse bằng RRF | `src/bonus_query_expansion.py` | Done — hit-rate 0.778 → 0.833 |
| Bonus — conversation memory | A/B 3 nhánh (không memory / lịch sử trong prompt / viết lại query) trên 7 hội thoại hai lượt | `src/bonus_conversation_memory.py`, `scripts/evaluate_memory.py` | Done — accuracy 0.714 → 0.857 |

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

2. **Quyết định:** Giữ nguyên chữ ký `retrieve()` theo contract; mọi nhánh
   bonus đi qua `run_retrieval()`.
   **Lý do/evidence:** Khi thêm tham số `reranker` và `query_variants` thẳng
   vào `retrieve()`, `test_public_function_signatures_are_stable` fail ngay.
   Cách dễ là sửa test cho khớp code — nhưng test đó chính là bằng chứng
   tuân thủ đặc tả của phần bắt buộc, đổi nó lấy điểm bonus là lỗ. Tôi tách
   `run_retrieval()` giữ toàn bộ logic, `retrieve()` thành wrapper bốn tham
   số. Cùng lý do: cross-encoder trả `retrieval_method="hybrid"` (giá trị
   contract cho phép) kèm khoá phụ `rerank_method`, thay vì thêm
   `"cross_encoder"` vào danh sách hợp lệ trong `contracts.py`.
   **Trade-off:** Có hai tên hàm cho một việc, và `scripts/` phải gọi
   `run_retrieval()` chứ không phải hàm quen thuộc hơn.

## Kiểm thử và kết quả

- Test đã dùng: `pytest -q` → 33/33 pass (15 contract + 5 acceptance + 13
  contract test cho nhánh bonus, chạy offline hoàn toàn: LLM provider, model
  reranker và vector store đều được monkeypatch);
  `python -m scripts.verify_golden_dataset` → 18/18 expected_context tìm thấy
  trong corpus; `python -m scripts.evaluate` cho A/B 5 config (18 câu × 4
  metric RAGAS mỗi config) và `python -m scripts.evaluate_memory` cho memory.
- Kết quả trước/sau: crawl cả trang cho ~40k ký tự/bài, trong đó ~95% là menu
  điều hướng lặp trên mọi URL. Sau khi khai báo selector nội dung
  (`div.bodytext`, `div.col-md-9`), mỗi bài còn 0.8k–8.6k ký tự nội dung thật.
- Lỗi đã phát hiện và cách xử lý: (1) BM25Okapi trả score 0 trên corpus nhỏ →
  chuyển BM25Plus; (2) ChromaDB từ chối `None` trong metadata → chuẩn hóa
  `url=None` thành chuỗi rỗng khi upsert; (3) title của tài liệu legal lấy từ
  dòng đầu PDF đều là "BỘ GIÁO DỤC VÀ ĐÀO TẠO", citation không phân biệt được
  văn bản → khai báo title tường minh trong `SOURCES`; (4) nhãn citation nếu
  đánh số SAU bước reorder chống lost-in-the-middle sẽ trỏ sai phần tử của
  `sources` (vì `sources` sort theo score, context thì không) → gán
  `citation_index` trước khi reorder.
- Lỗi phương pháp đo, phát hiện muộn: tôi đo latency bằng cách chạy xen kẽ
  các config theo một thứ tự **cố định**, và suốt ba lượt chạy dense-only
  luôn chậm hơn hybrid — dù nó làm strictly ít việc hơn. Nguyên nhân là
  config đứng đầu danh sách luôn là config chạy đầu tiên cho mỗi query mới
  nên vẫn gánh cold cache của riêng query đó. Xoay vòng thứ tự theo từng
  lượt thì hai con số về bằng nhau (28 ms), đúng như kỳ vọng.
- Một khẳng định tôi viết sai rồi phải sửa: trong bản nháp RESULT.md tôi ghi
  "cross-encoder không cứu được hp-02". Đối chiếu lại `eval_results.json`
  thì C và E đều HIT case đó. Case duy nhất trượt ở **mọi** config là hb-01.
  Tôi chuyển phần này sang tính thẳng từ dữ liệu thay vì viết tay.

## Điều còn hạn chế

- Hạn chế cụ thể: hybrid + RRF **kém hơn** dense-only trên context hit-rate
  (0.778 so với 0.889) và trên trung bình 4 metric RAGAS (0.763 so với 0.806;
  hybrid chỉ thắng ở context precision). Nguyên nhân là BM25 xếp hạng cao bản
  dịch tiếng Anh gần trùng nội dung (`hust-quy-che-dao-tao-tin-chi-en.md`), và
  RRF cho top-1 của BM25 trọng số `1/61` — đủ để đẩy chunk đúng ở hạng 3 của
  dense ra khỏi top-5. Cross-encoder che được triệu chứng (0.944) nhưng nguyên
  nhân trùng lặp dữ liệu thì vẫn còn nguyên.
- Cỡ mẫu nhỏ: golden dataset 18 câu, bộ follow-up 7 hội thoại. Chênh một case
  là 0.056 và 0.143 tương ứng. Đủ để chọn giữa hai config, chưa đủ để công bố
  con số như một kết luận tổng quát.
- PageIndex (task 8) chưa từng chạy với service thật vì không có
  `PAGEINDEX_API_KEY`; mới chỉ chứng minh được nhánh lỗi không làm sập
  pipeline và câu hỏi ngoài domain vẫn nhận safe refusal.
- Nếu có thêm thời gian, thay đổi đầu tiên: khử trùng lặp ở tầng corpus (loại
  bản dịch tiếng Anh hoặc gộp near-duplicate trước khi index), rồi đo lại xem
  RRF rẻ tiền có đủ dùng không — thay vì trả 557 ms mỗi query cho
  cross-encoder để bù một lỗi vốn nằm ở dữ liệu.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải
thích hoặc chạy lại trong buổi demo.

- Ngày: 2026-09-21
- Tên thành viên: Bùi Đức Vinh
