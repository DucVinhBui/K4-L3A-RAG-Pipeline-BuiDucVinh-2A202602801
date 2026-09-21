# Individual contribution report

## Thông tin

- Họ và tên: Bùi Đức Thông
- Mã học viên: 2A202602931
- Nhóm: L3A
- Repository/branch: `K4-L3A-RAG-Pipeline-BuiDucVinh-2A202602801` / `main`

## Cách nhóm làm việc

Nhóm thống nhất phương án theo từng module rồi giao cho một người chạy prompt
và commit tập trung (Bùi Đức Vinh), do chỉ tài khoản đó có quota. Vì vậy
`git log` chỉ hiện một tác giả; đóng góp của tôi nằm ở khâu quyết định phương
án, phân tích kết quả và kiểm thử, kê ở bảng dưới.

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| Chọn đề tài và phạm vi | Cùng nhóm chốt đề tài dịch vụ sinh viên HUST và phạm vi tài liệu | `data/`, `7a43099` | Done |
| Lexical search | Phát hiện BM25Okapi trả score 0 trên corpus của nhóm và chốt chuyển sang BM25Plus; chốt cách lọc "liên quan" bằng giao token thay cho `score > 0` | `src/task6_lexical_search.py`, `4a6aa06` | Done |
| Phân tích kết quả A/B | Đọc bảng 5 config, xác định hybrid+RRF thua dense-only và truy ra nguyên nhân là near-duplicate trong corpus chứ không phải lỗi thuật toán | `group_project/evaluation/RESULT.md`, `7a43099` | Done |
| Kiểm thử thủ công | Chạy chatbot, thử câu trong domain, câu ngoài domain và câu follow-up; đối chiếu nhãn `[Document N]` với danh sách nguồn hiển thị bên dưới | `app.py` | Done |
| Kiểm chứng trùng lặp corpus | Gỡ bản dịch EN, reindex, chạy lại A/B 5 config, so với RESULT.md cũ | `data/standardized/legal/`, `group_project/evaluation/RESULT.md` | Chưa bắt đầu |
| Test tay mở rộng | 15–20 câu tự nghĩ, kiểm citation và từng công tắc sidebar, ghi lại lỗi | `app.py` | Chưa bắt đầu |

## Quyết định kỹ thuật quan trọng

1. **Quyết định:** Dùng BM25Plus thay BM25Okapi cho lexical search. (Đây là
   quyết định nhóm bàn chung, phần tôi tham gia là chẩn đoán nguyên nhân;
   commit hiện thực là của Bùi Đức Vinh.)
   **Lý do/evidence:** idf của Okapi là `log(N-n+0.5) - log(n+0.5)`, bằng 0
   khi term xuất hiện ở khoảng một nửa số document. Corpus của nhóm chỉ 633
   chunk và rất đồng chủ đề, nên công thức này triệt tiêu đúng những term đặc
   trưng của domain — "học bổng", "tín chỉ", "cảnh báo" có mặt khắp nơi nên
   bị coi là vô nghĩa. Test contract với corpus 2 document cho toàn bộ score
   bằng 0. BM25Plus dùng `log((N+1)/n)` nên idf luôn dương.
   **Trade-off:** BM25Plus cộng một hằng số delta cho mọi document, kể cả
   document không chứa term nào trong query, nên không còn lọc "có liên quan"
   bằng `score > 0` được nữa; phải lọc bằng giao token giữa query và chunk.

2. **Quyết định:** Kết luận hybrid+RRF thua dense-only là do dữ liệu trùng
   lặp, không phải do RRF sai.
   **Lý do/evidence:** Hybrid+RRF kém hơn trên cả context hit-rate (0.778 so
   với 0.889) lẫn trung bình 4 metric RAGAS (0.763 so với 0.806), chỉ thắng ở
   context precision. Soi từng case trượt thì thấy BM25 xếp hạng cao
   `hust-quy-che-dao-tao-tin-chi-en.md` — bản dịch tiếng Anh gần trùng nội
   dung bản 2023. RRF cho top-1 của BM25 trọng số `1/61`, vừa đủ đẩy chunk
   đúng đang ở hạng 3 của dense ra khỏi top-5. Bằng chứng củng cố: cross-
   encoder dùng **đúng pool ứng viên đó** mà đạt 0.944, nghĩa là chunk đúng
   vẫn nằm trong pool, chỉ là RRF xếp sai — vì RRF chỉ nhìn thứ hạng, không
   nhìn nội dung nên không phân biệt được bản gốc với bản dịch.
   **Trade-off:** Kết luận này mới dựa trên quan sát và suy luận, chưa kiểm
   chứng bằng thực nghiệm khử trùng lặp rồi đo lại — đó là phần việc tôi đang
   nhận ở dòng "Kiểm chứng trùng lặp corpus" trong bảng trên.

## Kiểm thử và kết quả

- Test hoặc query tôi đã dùng: chạy `streamlit run app.py` và hỏi các câu
  trong domain (học bổng mấy mức, điều kiện cảnh báo học tập, học phí
  2025-2026) lẫn câu ngoài domain (giá vé tàu Hà Nội – Sài Gòn); đối chiếu
  từng nhãn `[Document N]` trong câu trả lời với danh sách nguồn ở expander
  bên dưới; bật tắt các công tắc cross-encoder, HyDE, nhớ hội thoại.
- Kết quả trước/sau: BM25Okapi cho toàn bộ score bằng 0 trên corpus test →
  BM25Plus cho score dương và xếp hạng có ý nghĩa. Câu ngoài domain nhận đúng
  safe refusal kể cả khi PageIndex lỗi do thiếu API key.
- Lỗi đã phát hiện và cách xử lý: BM25Okapi triệt tiêu idf trên corpus nhỏ
  đồng chủ đề → chuyển BM25Plus. Ngoài ra khi bật công tắc "Viết lại câu hỏi
  theo lịch sử", chất lượng trả lời không cải thiện mà còn kém hơn memory
  thường (0.714 so với 0.857) vì bản viết lại đôi khi đánh rơi cụm phân biệt
  → nhóm để công tắc này mặc định tắt thay vì bỏ hẳn.

## Điều còn hạn chế

- Một hạn chế cụ thể của phần tôi làm: kiểm thử thủ công của tôi mới dùng lại
  các câu có sẵn trong golden dataset và câu hỏi mẫu, tức là những câu hệ
  thống đã được tối ưu để trả lời tốt. Chưa có bộ câu do người ngoài nhóm tự
  nghĩ, nên con số "chạy tốt" hiện tại lạc quan hơn thực tế.
- Nếu có thêm thời gian, thay đổi đầu tiên tôi sẽ thực hiện: chạy thực nghiệm
  khử trùng lặp corpus rồi đo lại A/B. Nếu hybrid+RRF vượt được dense-only
  sau khi khử, thì kết luận là lỗi nằm ở dữ liệu và khoản 557 ms mỗi query
  trả cho cross-encoder là tiền trả thừa.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải
thích hoặc chạy lại trong buổi demo.

- Ngày: 2026-09-21
- Tên thành viên: Bùi Đức Thông
