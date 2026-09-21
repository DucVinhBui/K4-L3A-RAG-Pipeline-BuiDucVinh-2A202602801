# RAG evaluation results

## Run information

| Field                              | Value |
| ---------------------------------- | ----- |
| Evaluation date                    | 2026-09-21 |
| Framework and version              | RAGAS 0.4.3 |
| Evaluator model                    | gpt-4o-mini |
| Generator model                    | openai/gpt-4o-mini |
| Embedding model                    | BAAI/bge-m3 (chunk 500/50) |
| Corpus version/commit              | 13 docs / 633 chunks @ `6a2a2d4` |
| Golden dataset size                | 18 |
| `top_k`                            | 5 |
| Fallback threshold and calibration | 0.59 — `scripts/calibrate_threshold.py`: 10 query in-domain (min 0.682) tách sạch khỏi 8 query out-of-domain (max 0.488) |

## Configurations

- **Config A — dense-only:** `retrieve(query, top_k=5, use_reranking=False)` — chỉ ChromaDB cosine trên bge-m3, lấy thẳng top-5 dense.
- **Config B — hybrid + RRF:** `retrieve(query, top_k=5, use_reranking=True)` — dense top-10 và BM25Plus top-10 trên cùng corpus, fuse một lần bằng RRF `1/(60+rank)`, lấy top-5.

Hai config dùng cùng golden dataset, generator, evaluator, prompt và `top_k`; chỉ thay retrieval strategy.

## Overall scores

| Metric            | Config A | Config B | Delta B−A |
| ----------------- | -------: | -------: | --------: |
| Faithfulness      |    0.870 |    0.833 |    -0.037 |
| Answer relevance  |    0.539 |    0.529 |    -0.010 |
| Context recall    |    0.861 |    0.800 |    -0.061 |
| Context precision |    0.886 |    0.920 |    +0.033 |
| **Average**       |    0.789 |    0.771 |    -0.019 |

Chỉ số tất định đo kèm, không cần LLM-judge:

| Metric | Config A | Config B | Delta B−A |
| --- | ---: | ---: | ---: |
| Context hit-rate | 0.889 | 0.778 | -0.111 |
| Median retrieval (ms) | 31 | 30 | -1 |
| Mean generation (s) | 1.60 | 1.28 | -0.32 |

*Context hit-rate = tỉ lệ câu hỏi mà `expected_context` xuất hiện nguyên văn trong contexts đã truy hồi. Retrieval và generation được bấm giờ riêng: gộp lại thì latency bị LLM call chi phối. Retrieval latency lấy trung vị của benchmark chạy xen kẽ A/B 3 lượt — đo trong lượt chính thì config chạy trước gánh phần làm nóng cache và cho số phi lý.*

## A/B comparison

- Cấu hình tốt hơn: **Config A — dense-only**
- Evidence: context hit-rate 0.889 (A) so với 0.778 (B), delta -0.111; trung bình 4 metric RAGAS 0.789 (A) so với 0.771 (B).
- Kết quả đi ngược kỳ vọng thông thường. Nguyên nhân đã truy ra: corpus chứa hai cặp tài liệu gần trùng nội dung (bản quy chế đào tạo tiếng Anh so với bản tiếng Việt, và hai bài viết cùng mô tả 3 mức học bổng). BM25 xếp hạng 1 cho bản gần trùng, và RRF cho top-1 của mỗi list trọng số `1/61` — lớn hơn hạng 3 của dense (`1/63`) — nên chunk đúng bị đẩy khỏi top-5. Vấn đề nằm ở trùng lặp dữ liệu, không phải ở công thức RRF.
- Trade-off về latency/cost: B thêm một lượt BM25 trên 633 chunk, retrieval tốn 30 ms so với 31 ms của A (-1 ms/query, trung vị) vì BM25 index được cache sau lần build đầu. B không tốn thêm API call; generation (1.60s so với 1.28s) chênh nhau do độ dài context và độ trễ phía OpenAI, không phải do retrieval strategy. Chi phí thêm của B là không đáng kể — lý do chưa chọn B là chất lượng, không phải giá.

## Worst performers

|   # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause |
| --: | -------- | ------ | -----------: | --------: | -----: | --------: | ------------- | ---------- |
|   1 | Học bổng khuyến khích học tập của Bách khoa Hà Nội có mấy mức? | B | 0.000 | 0.511 | 0.000 | 0.750 | retrieval + generation | Cụm 'của Bách khoa Hà Nội' trong câu hỏi kéo retrieval về các chunk giới thiệu chung (liệt kê các loại học bổng) thay vì chunk liệt kê 3 mức A/B/C. Model vẫn trả lời 'có 3 mức' nhưng context không nêu đủ ba mức, nên faithfulness bị chấm 0. Bỏ cụm tên trường khỏi câu hỏi thì retrieval lấy đúng chunk và câu trả lời liệt kê đủ A/B/C. |
|   2 | Học phí năm học 2025-2026 được tính dựa trên cơ sở nào? | B | 0.833 | 0.551 | 0.000 | 0.700 | data/chunking | Câu định nghĩa cách tính học phí nằm vắt qua ranh giới chunk-3/chunk-4 của quyết định học phí; retrieval lấy chunk-3 (phần căn cứ pháp lý) nên thiếu đúng mệnh đề trả lời. Overlap 50 ký tự không đủ cho đoạn văn bản hành chính dài. |
|   3 | Điều kiện để sinh viên được xét công nhận tốt nghiệp là gì? | B | 1.000 | 0.466 | 0.400 | 0.804 | retrieval | BM25 xếp hạng 1 cho bản dịch tiếng Anh gần trùng (hust-quy-che-dao-tao-tin-chi-en.md). RRF cho top-1 của BM25 trọng số 1/61, đẩy chunk đúng ở hạng 3 của dense (1/63) ra khỏi top-5. |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| -------: | ------ | ------------------------------ | --------------- | ------------- |
|        1 | Khử trùng lặp ở tầng corpus: loại bản dịch tiếng Anh `hust-quy-che-dao-tao-tin-chi-en.md` hoặc gộp near-duplicate trước khi index | 2/3 worst performer là do BM25 xếp hạng 1 cho bản gần trùng rồi RRF đẩy chunk đúng khỏi top-5 | Hybrid hết bị phạt bởi trùng lặp; kỳ vọng context hit-rate của B vượt 0.889 của A | Chạy lại `python -m scripts.evaluate` và so delta B−A |
|        2 | Tăng `CHUNK_OVERLAP` từ 50 lên ~150 cho tài liệu legal, hoặc chunk theo ranh giới Điều/Khoản thay vì ký tự | hp-02 hỏng vì câu định nghĩa cách tính học phí nằm vắt qua ranh giới chunk-3/chunk-4 | Giảm lỗi mất mệnh đề ở văn bản hành chính dài | `python -m scripts.verify_golden_dataset` rồi `scripts.evaluate`, theo dõi riêng nhóm câu hỏi `category=học phí` |
|        3 | Bổ sung chỉ số đánh giá theo nội dung tương đương thay vì khớp chuỗi tuyệt đối | hb-01 bị chấm miss dù context truy hồi được (article_05) trả lời đúng y hệt ground truth trích từ article_06 | Hit-rate phản ánh đúng chất lượng, tránh tối ưu nhầm hướng | Đối chiếu hit-rate với `context_recall` của RAGAS trên cùng lần chạy |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| ---------- | -------- | -----------: | -----------------: | ---------- |
| Conversation memory cho follow-up (toggle trong `app.py`, 2 lượt gần nhất ghép vào user message) | Không có memory | chưa đo định lượng | +0 API call, prompt dài thêm ~400 token/lượt | Chạy được, demo bằng câu hỏi nối tiếp; chưa có A/B định lượng |
| UI citation/source highlighting (`app.py` hiển thị `[Document N]`, retrieval_method, score và link nguồn cho từng chunk) | UI chỉ có answer | không áp dụng | +0 | Chạy được, kiểm chứng bằng demo |

