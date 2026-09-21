# RAG evaluation results

## Run information

| Field                              | Value |
| ---------------------------------- | ----- |
| Evaluation date                    | 2026-09-21 |
| Framework and version              | RAGAS 0.4.3 |
| Evaluator model                    | gpt-4o-mini |
| Generator model                    | openai/gpt-4o-mini |
| Embedding model                    | BAAI/bge-m3 (chunk 500/50) |
| Corpus version/commit              | 13 docs / 633 chunks @ `4c82fcb` |
| Golden dataset size                | 18 |
| `top_k`                            | 5 |
| Fallback threshold and calibration | 0.59 — `scripts/calibrate_threshold.py`: 10 query in-domain (min 0.682) tách sạch khỏi 8 query out-of-domain (max 0.488) |

## Configurations

- **Config A — dense-only:** `run_retrieval(query, top_k=5, use_reranking=False)` — chỉ ChromaDB cosine trên bge-m3, lấy thẳng top-5 dense.
- **Config B — hybrid + RRF:** `run_retrieval(query, top_k=5, use_reranking=True)` — dense top-10 và BM25Plus top-10 trên cùng corpus, fuse một lần bằng RRF `1/(60+rank)`, lấy top-5. **Baseline của mọi config bonus.**
- **Config C — hybrid + cross-encoder:** `run_retrieval(..., reranker="cross_encoder")` — cùng pool ứng viên như B nhưng chấm lại từng cặp (query, chunk) bằng `BAAI/bge-reranker-v2-m3` thay vì fuse theo thứ hạng.
- **Config D — HyDE + hybrid + RRF:** `run_retrieval(..., query_variants=expand_query(query))` — LLM viết một đoạn văn giả định trả lời câu hỏi, dense search thêm trên đoạn đó, rồi fuse cả ba ranked list bằng RRF như B.
- **Config E — HyDE + hybrid + cross-encoder:** Gộp C và D: pool ứng viên có thêm nhánh HyDE, chấm lại bằng cross-encoder.

Mọi config dùng chung golden dataset, generator, evaluator, prompt và `top_k`; chỉ thay retrieval strategy. A/B chính theo yêu cầu bài là **A so với B**; các config bonus C/D/E lấy **B** làm mốc vì mỗi config chỉ đổi đúng một thành phần của B, nên chênh lệch quy được về đúng thành phần đó.

## Overall scores

| Metric            | Config A | Config B | Delta B−A |
| ----------------- | -------: | -------: | --------: |
| Faithfulness      |    0.880 |    0.824 |    -0.056 |
| Answer relevance  |    0.530 |    0.525 |    -0.006 |
| Context recall    |    0.917 |    0.800 |    -0.117 |
| Context precision |    0.897 |    0.903 |    +0.006 |
| **Average**       |    0.806 |    0.763 |    -0.043 |

Chỉ số tất định đo kèm, không cần LLM-judge:

| Metric | Config A | Config B | Delta B−A |
| --- | ---: | ---: | ---: |
| Context hit-rate | 0.889 | 0.778 | -0.111 |
| Median retrieval (ms) | 28 | 28 | +0 |
| Mean generation (s) | 1.43 | 1.33 | -0.10 |

*Context hit-rate = tỉ lệ câu hỏi mà `expected_context` xuất hiện nguyên văn trong contexts đã truy hồi. Retrieval và generation được bấm giờ riêng: gộp lại thì latency bị LLM call chi phối. Retrieval latency lấy trung vị của benchmark riêng, chạy 5 lượt và XOAY VÒNG thứ tự config mỗi lượt. Hai lần đo trước đều sai: đo trong lượt chính thì config chạy trước gánh phần làm nóng cache, còn xen kẽ theo thứ tự cố định thì config đứng đầu vẫn gánh cold cache của từng query — cho ra dense-only chậm hơn hybrid suốt ba lượt, dù nó làm strictly ít việc hơn.*

### Toàn bộ config

| Metric | A | B | C | D | E |
| --- | ---: | ---: | ---: | ---: | ---: |
| Faithfulness | 0.880 | 0.824 | 0.917 | 0.889 | 0.861 |
| Answer relevance | 0.530 | 0.525 | 0.535 | 0.536 | 0.529 |
| Context recall | 0.917 | 0.800 | 0.917 | 0.861 | 0.898 |
| Context precision | 0.897 | 0.903 | 0.952 | 0.936 | 0.941 |
| **Average** | 0.806 | 0.763 | 0.830 | 0.805 | 0.807 |
| Context hit-rate | 0.889 | 0.778 | 0.944 | 0.833 | 0.944 |
| Median retrieval (ms) | 28 | 28 | 557 | 68 | 718 |
| Mean expansion (s) | 0.00 | 0.00 | 0.00 | 1.64 | 1.64 |
| Mean generation (s) | 1.43 | 1.33 | 1.38 | 1.36 | 1.36 |

Config tốt nhất trên toàn bộ phép đo: **Config C — hybrid + cross-encoder** (trung bình 0.830, hit-rate 0.944). *Mean expansion* là chi phí LLM sinh đoạn văn HyDE, tách khỏi latency retrieval vì đó là network call chứ không phải công việc của retrieval.

## A/B comparison

- Cấu hình tốt hơn: **Config A — dense-only**
- Evidence: context hit-rate 0.889 (A) so với 0.778 (B), delta -0.111; trung bình 4 metric RAGAS 0.806 (A) so với 0.763 (B).
- Kết quả đi ngược kỳ vọng thông thường. Nguyên nhân đã truy ra: corpus chứa hai cặp tài liệu gần trùng nội dung (bản quy chế đào tạo tiếng Anh so với bản tiếng Việt, và hai bài viết cùng mô tả 3 mức học bổng). BM25 xếp hạng 1 cho bản gần trùng, và RRF cho top-1 của mỗi list trọng số `1/61` — lớn hơn hạng 3 của dense (`1/63`) — nên chunk đúng bị đẩy khỏi top-5. Vấn đề nằm ở trùng lặp dữ liệu, không phải ở công thức RRF.
- Chẩn đoán trên được kiểm chứng bằng Config C: giữ nguyên pool ứng viên của B, chỉ thay bước fuse bằng cross-encoder đọc nội dung, thì hit-rate lên 0.944 — cao hơn cả A. Đúng là khâu 'tin vào thứ hạng của BM25' gây lỗi, không phải khâu lấy ứng viên.
- Trade-off về latency/cost: B thêm một lượt BM25 trên 633 chunk, retrieval tốn 28 ms so với 28 ms của A (+0 ms/query, trung vị) vì BM25 index được cache sau lần build đầu. B không tốn thêm API call; generation (1.43s so với 1.33s) chênh nhau do độ dài context và độ trễ phía OpenAI, không phải do retrieval strategy. Chi phí thêm của B là không đáng kể — lý do chưa chọn B là chất lượng, không phải giá.

## Worst performers

|   # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause |
| --: | -------- | ------ | -----------: | --------: | -----: | --------: | ------------- | ---------- |
|   1 | Học bổng khuyến khích học tập của Bách khoa Hà Nội có mấy mức? | B | 0.000 | 0.511 | 0.000 | 0.750 | retrieval + generation | Cụm 'của Bách khoa Hà Nội' trong câu hỏi kéo retrieval về các chunk giới thiệu chung (liệt kê các loại học bổng) thay vì chunk liệt kê 3 mức A/B/C. Model vẫn trả lời 'có 3 mức' nhưng context không nêu đủ ba mức, nên faithfulness bị chấm 0. Bỏ cụm tên trường khỏi câu hỏi thì retrieval lấy đúng chunk và câu trả lời liệt kê đủ A/B/C. |
|   2 | Học phí năm học 2025-2026 được tính dựa trên cơ sở nào? | A | 1.000 | 0.551 | 0.000 | 0.500 | data/chunking | Câu định nghĩa cách tính học phí nằm vắt qua ranh giới chunk-3/chunk-4 của quyết định học phí; retrieval lấy chunk-3 (phần căn cứ pháp lý) nên thiếu đúng mệnh đề trả lời. Overlap 50 ký tự không đủ cho đoạn văn bản hành chính dài. |
|   3 | Điều kiện để sinh viên được xét công nhận tốt nghiệp là gì? | D | 1.000 | 0.467 | 0.000 | 1.000 | retrieval | BM25 xếp hạng 1 cho bản dịch tiếng Anh gần trùng (hust-quy-che-dao-tao-tin-chi-en.md). RRF cho top-1 của BM25 trọng số 1/61, đẩy chunk đúng ở hạng 3 của dense (1/63) ra khỏi top-5. |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| -------: | ------ | ------------------------------ | --------------- | ------------- |
|        1 | Chuyển retrieval mặc định sang **Config C — hybrid + cross-encoder** (`reranker="cross_encoder"`) thay cho RRF | Đã đo, không phải phỏng đoán: cùng pool ứng viên của B, chỉ thay bước fuse, hit-rate 0.778 → 0.944 và trung bình RAGAS 0.763 → 0.830 | Đã đạt: +0.167 hit-rate. Việc còn lại là quyết định có chấp nhận latency không | Latency retrieval tăng 28 ms → 557 ms; chạy `python -m scripts.evaluate --benchmark-only` trên phần cứng đích |
|        2 | Khử trùng lặp ở tầng corpus: loại bản dịch tiếng Anh `hust-quy-che-dao-tao-tin-chi-en.md` hoặc gộp near-duplicate trước khi index | 2/3 worst performer là do BM25 xếp hạng 1 cho bản gần trùng rồi RRF đẩy chunk đúng khỏi top-5. Cross-encoder đã che được triệu chứng này, nhưng nguyên nhân vẫn còn và vẫn tốn chỗ trong pool ứng viên | Hybrid hết bị phạt bởi trùng lặp; RRF rẻ có thể đủ dùng, khỏi trả giá latency của cross-encoder | Chạy lại `python -m scripts.evaluate` và so delta B−A |
|        3 | Tăng `CHUNK_OVERLAP` từ 50 lên ~150 cho tài liệu legal, hoặc chunk theo ranh giới Điều/Khoản thay vì ký tự | Câu định nghĩa cách tính học phí (hp-02) nằm vắt qua ranh giới chunk-3/chunk-4. Cross-encoder tình cờ cứu được case này, nhưng bằng cách xếp hạng lại chứ không phải bằng cách làm chunk đúng tồn tại — khâu chunking vẫn đang cắt mất mệnh đề trả lời | Giảm lỗi mất mệnh đề ở văn bản hành chính dài, và không phải dựa vào reranker để bù | `python -m scripts.verify_golden_dataset` rồi `scripts.evaluate`, theo dõi riêng nhóm câu hỏi `category=học phí` |
|        4 | Xem lại cách đặt câu hỏi và ranh giới tài liệu cho hb-01 — case miss ở **mọi** config đã thử | 1/18 case không có retrieval strategy nào lấy đúng context: dense, hybrid, cross-encoder và HyDE đều trượt. Đây là giới hạn của dữ liệu hoặc của ground truth, không phải của thuật toán xếp hạng | Tách được phần lỗi còn lại thành lỗi dữ liệu thay vì tiếp tục tinh chỉnh retrieval vô ích | Đối chiếu `expected_context` của case đó với chunk thật trong `chroma_db` bằng `scripts/verify_golden_dataset.py` |
|        5 | Bổ sung chỉ số đánh giá theo nội dung tương đương thay vì khớp chuỗi tuyệt đối | hb-01 bị chấm miss dù context truy hồi được (article_05) trả lời đúng y hệt ground truth trích từ article_06 | Hit-rate phản ánh đúng chất lượng, tránh tối ưu nhầm hướng | Đối chiếu hit-rate với `context_recall` của RAGAS trên cùng lần chạy |

*Các case dense-only trượt nhưng một config khác lấy được: hp-02. Chi tiết từng case trong `eval_results.json`.*

## Bonus experiments

Mọi config bonus so với **Config B — hybrid + RRF** vì mỗi config chỉ đổi đúng một thành phần của B.

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| ---------- | -------- | -----------: | -----------------: | ---------- |
| Cross-encoder rerank (`BAAI/bge-reranker-v2-m3`) thay RRF, cùng pool ứng viên | Config B | hit-rate +0.167, RAGAS avg +0.067 | +528 ms/query, +0 API call, nạp ~2.2GB weight một lần | Cải thiện rõ trên mọi metric. RRF chỉ nhìn thứ hạng nên tin theo BM25 khi BM25 xếp hạng 1 cho bản gần trùng; cross-encoder đọc nội dung nên không mắc lỗi đó. Đánh đổi là latency retrieval tăng gần 20 lần. |
| HyDE query expansion (`scripts`/`src/bonus_query_expansion.py`) cộng vào B | Config B | hit-rate +0.056, RAGAS avg +0.043 | +40 ms/query, +1.64s LLM, +1 LLM call/query | Có cải thiện so với B nhưng nhỏ hơn cross-encoder và phải trả thêm một LLM call. Đoạn văn giả định kéo query về đúng thể loại văn bản hành chính, bù được một phần lỗi lệch thể loại câu hỏi/đoạn văn. |
| Gộp cả hai: HyDE + cross-encoder | Config B | hit-rate +0.167, RAGAS avg +0.044 | +689 ms/query, +1.64s LLM, +1 LLM call/query, nạp ~2.2GB weight | Hai cải tiến không cộng dồn. Cross-encoder đã sửa gần hết phần lỗi mà HyDE nhắm tới, nên thêm HyDE lên trên nó gần như không đổi kết quả trong khi vẫn phải trả thêm một LLM call mỗi câu. |
| Conversation memory: ghép lịch sử vào prompt (`app.py`) | Không memory, cùng 7 hội thoại hai lượt | follow-up accuracy +0.143 (0.714 → 0.857) | +0 API call, prompt dài thêm ~400 token/lượt | **Có cải thiện.** Context hit-rate đã là 1.000 khi chưa có memory, nên chỗ hỏng không nằm ở retrieval mà ở generation: model có đủ context nhưng không nối được đại từ hồi chỉ với chủ thể, và trả về refusal. Lịch sử trong prompt sửa đúng chỗ đó. |
| Conversation memory: viết lại câu hỏi theo lịch sử trước khi retrieve (`src/bonus_conversation_memory.py`) | Memory trong prompt | follow-up accuracy -0.143 (0.857 → 0.714), context hit-rate -0.143 | +1 LLM call/query (~0.94s) | **Không dùng.** Sửa được fu-01 nhưng làm hỏng fu-07: bản viết lại đánh rơi cụm 'để được xét công nhận tốt nghiệp', retrieval mất chunk đúng và câu trả lời thành refusal. Đổi một lỗi ở prompt (context vẫn còn) lấy một lỗi ở retrieval (không còn gì để cứu). Giữ làm toggle, mặc định tắt. |
| UI citation/source highlighting (`app.py` hiển thị `[Document N]`, retrieval_method, score và link nguồn cho từng chunk) | UI chỉ có answer | không áp dụng | +0 | Chạy được, kiểm chứng bằng demo |

*Bộ follow-up có 7 hội thoại — chênh một case là 0.143. Đủ để quyết định bật hay tắt một toggle, chưa đủ để kết luận tổng quát. Mọi câu lượt 2 đều là ellipsis hoặc có đại từ hồi chỉ; thước đo là khớp chuỗi dữ kiện, không dùng LLM-judge. Chi tiết từng case: `group_project/evaluation/memory_results.json`.*

