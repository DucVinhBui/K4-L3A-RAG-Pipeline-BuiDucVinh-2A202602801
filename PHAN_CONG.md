# Phân công phần việc còn lại

Ba việc dưới đây độc lập nhau, không ai chặn ai, mỗi việc ước chừng 1–2 giờ
và đều kết thúc bằng một commit + một con số đối chiếu được. Làm xong thì
điền thẳng vào bảng trong `reports/<mssv>-<tên>.md` của mình.

---

## Đinh Công Tú (2A202602479) — bổ sung nguồn tin tức

**Việc:** Thêm 3–5 bài từ `ctt.hust.edu.vn` vào corpus. Chủ đề còn thiếu:
ký túc xá, bảo hiểm y tế sinh viên, thủ tục cấp bảng điểm / giấy xác nhận.

**Cách làm:**
1. Thêm URL vào danh sách nguồn trong `src/task2_crawl_news.py`.
2. `python -m src.task2_crawl_news` — kiểm tra selector nội dung có bắt đúng
   không. Bài crawl đúng dài 0.8k–8.6k ký tự; nếu ra ~40k thì đang dính menu
   điều hướng, phải khai báo selector riêng cho trang đó.
3. `python -m src.task3_convert_markdown` rồi `python -m src.task4_chunking_indexing`.

**Số cần ghi vào báo cáo:** số chunk trước/sau (hiện tại 633), số ký tự nội
dung thật mỗi bài mới.

---

## Đỗ Phúc Hưng (2A202602762) — mở rộng golden dataset

**Việc:** Nâng `group_project/evaluation/golden_dataset.json` từ 18 lên ~30
câu. Cỡ mẫu hiện tại quá nhỏ: chênh đúng một case đã là 0.056.

**Cách làm:**
1. Mở `data/standardized/` đọc tài liệu, soạn câu hỏi có đáp án là một con số
   hoặc một tên riêng rõ ràng — tránh đáp án chung chung kiểu "không", vì
   script chấm bằng string containment sẽ báo đúng nhầm.
2. Theo đúng schema các câu đã có (`question`, `expected_answer`,
   `expected_context`).
3. `python -m scripts.verify_golden_dataset` phải pass 100% mới được dừng.

**Số cần ghi vào báo cáo:** số câu trước/sau, tỉ lệ verify pass, và câu nào
phải sửa lại vì không tìm thấy trong chunk đã index.

---

## Bùi Đức Thông (2A202602931) — kiểm chứng giả thuyết trùng lặp corpus

**Việc:** RESULT.md đang ghi hybrid+RRF *thua* dense-only (0.763 so với 0.806)
và nghi nguyên nhân là `data/standardized/legal/hust-quy-che-dao-tao-tin-chi-en.md`
— bản dịch tiếng Anh gần trùng nội dung bản 2023, bị BM25 đẩy lên top. Chưa ai
kiểm chứng. Việc của bạn là kiểm chứng.

**Cách làm:**
1. Gỡ file đó khỏi corpus, chạy lại `python -m src.task4_chunking_indexing`.
2. `python -m scripts.evaluate` (chạy đủ, không dùng `--benchmark-only`).
3. So bảng 5 config mới với bảng cũ trong `group_project/evaluation/RESULT.md`.

**Số cần ghi vào báo cáo:** hybrid+RRF có vượt được dense-only sau khi khử
trùng lặp không. Nếu có, kết luận là lỗi nằm ở dữ liệu chứ không ở thuật toán
— và cross-encoder 557 ms/query là khoản trả thừa. Nếu không, giả thuyết trong
RESULT.md sai và phải sửa lại.

> Nhớ backup `eval_results.json` cũ trước khi chạy, nếu không mất bảng đối chiếu.

---

# Đợt 2 — kiểm chứng phần đã có

Ba việc dưới đây không viết code mới mà soi lại phần đã làm. Người soi không
phải người viết thì mới tìm ra lỗi — và rubric có 5 điểm cho "README, khả năng
chạy lại" mà tới giờ chưa ai kiểm ngoài chính tác giả.

## Đinh Công Tú — dựng lại từ clone sạch

Clone repo về một thư mục mới, làm **đúng từng dòng** trong README, không dùng
kiến thức sẵn có, không tự đoán bước thiếu. Mỗi lần khựng lại thì ghi lại chỗ
khựng.

Cần trả lời: từ máy trắng đến lúc `streamlit run app.py` lên được mất bao lâu,
và README thiếu những bước nào. Sửa README bằng đúng những gì mình vấp phải.
Đó là commit của bạn.

## Đỗ Phúc Hưng — đối chiếu golden dataset với văn bản gốc

18 câu trong `golden_dataset.json` mới được script xác minh là *có xuất hiện
trong chunk đã index* — chưa ai xác minh đáp án **đúng** so với văn bản gốc.
Mở từng câu, tra ngược về `data/standardized/`, đối chiếu con số.

Cần trả lời: có câu nào đáp án sai, mơ hồ, hoặc có hai cách hiểu không. Đặc
biệt soi `hb-01` — case duy nhất trượt ở cả 5 config, rất có thể lỗi ở câu hỏi
chứ không ở retrieval.

## Bùi Đức Thông — test tay chatbot

Chạy `streamlit run app.py`, hỏi 15–20 câu tự nghĩ (đừng lấy lại câu trong
golden dataset), trong đó cố tình có vài câu ngoài domain và vài câu hỏi
follow-up kiểu "thế còn mức 2 thì sao".

Với mỗi câu ghi lại: trả lời đúng/sai, citation `[Document N]` có trỏ đúng
nguồn trong danh sách bên dưới không, câu ngoài domain có từ chối an toàn
không. Bật/tắt thử các công tắc ở sidebar (cross-encoder, HyDE, nhớ hội thoại)
xem có công tắc nào làm hỏng câu trả lời không.

Cần trả lời: tỉ lệ đúng trên bộ câu tự nghĩ, và danh sách lỗi cụ thể tìm được.
