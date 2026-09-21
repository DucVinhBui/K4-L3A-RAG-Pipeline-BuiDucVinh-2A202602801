# Individual contribution report

## Thông tin

- Họ và tên: Đỗ Phúc Hưng
- Mã học viên: 2A202602762
- Nhóm: L3A
- Repository/branch: `K4-L3A-RAG-Pipeline-BuiDucVinh-2A202602801` / `main`

## Cách nhóm làm việc

Nhóm thống nhất phương án theo từng module rồi giao cho một người chạy prompt
và commit tập trung (Bùi Đức Vinh), do chỉ tài khoản đó có quota. Vì vậy
`git log` chỉ hiện một tác giả; đóng góp của tôi nằm ở khâu quyết định phương
án, chuẩn bị dữ liệu đánh giá và kiểm thử, kê ở bảng dưới.

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| Chọn đề tài và phạm vi | Cùng nhóm chốt đề tài dịch vụ sinh viên HUST và phạm vi tài liệu | `data/`, `7a43099` | Done |
| Golden dataset | Soạn 18 câu hỏi trích từ corpus kèm `expected_answer` và `expected_context`; yêu cầu mọi đáp án phải là con số hoặc tên riêng tra ngược được về văn bản gốc | `group_project/evaluation/golden_dataset.json`, `7a43099` | Done |
| Bộ follow-up | Soạn 7 hội thoại hai lượt, mỗi lượt 2 đều là câu tỉnh lược hoặc dùng đại từ hồi chỉ nên bắt buộc phải có lịch sử mới hiểu; kèm trường `why_needs_history` giải thích từng case | `group_project/evaluation/followup_dataset.json`, `856db30` | Done |
| Thiết kế A/B | Chốt bộ metric (4 metric RAGAS + context hit-rate deterministic) và 5 config đem so: dense-only, hybrid+RRF, cross-encoder, HyDE, HyDE+cross-encoder | `scripts/evaluate.py`, `4bf3194` | Done |
| Mở rộng golden dataset | Nâng 18 → ~30 câu, verify 100% có thật trong chunk đã index | `group_project/evaluation/golden_dataset.json` | Chưa bắt đầu |

## Quyết định kỹ thuật quan trọng

1. **Quyết định:** Đo bằng cả metric LLM-judge lẫn một metric deterministic
   (context hit-rate bằng string containment), không chỉ dựa vào RAGAS.
   **Lý do/evidence:** Answer relevance do LLM chấm ra 0.525–0.536 ở **cả
   năm** config — biên độ 0.011, không tách nổi config nào hơn config nào.
   Trong khi đó context hit-rate tách rất rõ: 0.778 cho hybrid+RRF so với
   0.944 cho cross-encoder. Nếu chỉ nhìn RAGAS thì kết luận của cả bài là
   "năm config như nhau", tức là không kết luận được gì.
   **Trade-off:** String containment chấm đúng/sai theo mặt chữ nên chỉ dùng
   được cho câu hỏi một dữ kiện; câu cần diễn giải thì nó vô dụng, và nó
   không thay được LLM-judge ở faithfulness.

2. **Quyết định:** Bắt buộc mọi đáp án trong bộ follow-up phải là số hoặc
   tên riêng, loại hết đáp án là từ phổ thông.
   **Lý do/evidence:** Bản đầu của bộ follow-up có case mà `expected_any`
   chứa chuỗi `"không"`. Chuỗi này xuất hiện trong gần như mọi câu tiếng
   Việt, nên hàm chấm báo đúng cho cả những lần mô hình trả lời sai — A/B
   memory ra chênh lệch +0.000, một kết quả trông như "memory vô dụng" nhưng
   thật ra là thước đo hỏng. Dựng lại 7 case với đáp án dứt khoát ("1,5",
   "16", "C7", "2 mức", "bằng 4", "26/9", "2,0") thì phép đo mới chạy: không
   memory 0.714, memory trong prompt 0.857.
   **Trade-off:** Ràng buộc này loại mất loại câu hỏi mở — vốn là loại người
   dùng thật hay hỏi nhất — nên bộ đánh giá dễ hơn thực tế sử dụng.

## Kiểm thử và kết quả

- Test hoặc query tôi đã dùng: `python -m scripts.verify_golden_dataset` để
  chứng minh mọi `expected_context` thật sự nằm trong chunk đã index;
  `python -m scripts.evaluate` chạy 5 config × 18 câu × 4 metric RAGAS;
  `python -m scripts.evaluate_memory` chạy 3 nhánh trên 7 hội thoại.
- Kết quả trước/sau: verify 18/18 pass. A/B cho trung bình 4 metric:
  dense-only 0.806, hybrid+RRF 0.763, cross-encoder 0.830, HyDE 0.805,
  HyDE+cross-encoder 0.807. A/B memory: 0.714 / 0.857 / 0.714.
- Lỗi đã phát hiện và cách xử lý: (1) đáp án `"không"` trong bộ follow-up gây
  báo đúng nhầm, làm A/B memory ra +0.000 giả → dựng lại bộ với đáp án là số
  hoặc tên riêng; (2) giả thuyết ban đầu của nhóm là memory giúp ở khâu
  retrieval, nhưng đo ra context hit-rate đã là 1.000 ngay cả khi không có
  memory — nút thắt nằm ở khâu sinh câu trả lời, không phải truy hồi. Kết
  luận trong báo cáo đã sửa lại theo số đo thay vì giữ giả thuyết.

## Điều còn hạn chế

- Một hạn chế cụ thể của phần tôi làm: cỡ mẫu quá nhỏ. Golden dataset 18 câu
  nên chênh đúng một case đã là 0.056; bộ follow-up 7 hội thoại thì một case
  là 0.143. Đủ để chọn giữa hai config, chưa đủ để công bố con số như kết
  luận tổng quát — khoảng cách 0.806 với 0.830 hoàn toàn có thể là nhiễu.
- Nếu có thêm thời gian, thay đổi đầu tiên tôi sẽ thực hiện: nâng golden
  dataset lên 30+ câu phủ đều cả 4 văn bản pháp quy lẫn 9 bài news, và điều
  tra `hb-01` — case duy nhất trượt ở **cả 5** config, nhiều khả năng lỗi nằm
  ở chính câu hỏi tôi soạn chứ không ở retrieval.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải
thích hoặc chạy lại trong buổi demo.

- Ngày: 2026-09-21
- Tên thành viên: Đỗ Phúc Hưng
