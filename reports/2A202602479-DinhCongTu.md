**Individual contribution report**  
**Thông tin**  
- Họ và tên: Đinh Công Tú  
- Mã học viên: 2A202602479  
- Nhóm: L3A  
- Tài khoản GitHub: TuTune04  
- Repository/branch: DucVinhBui/K4-L3A-RAG-Pipeline-BuiDucVinh-2A202602801 / main  
- Phạm vi phụ trách: rà soát dữ liệu, phân tích kết quả retrieval và kiểm tra khả năng tái lập  
**Phần việc đã thực hiện**  
| | | | |  
|-|-|-|-|  
| **Module/deliverable** | **Việc tôi trực tiếp thực hiện** | **Bằng chứng đối chiếu** | **Trạng thái** |   
| Rà soát corpus và độ bao phủ chủ đề | Kiểm kê corpus hiện tại gồm 4 PDF chính sách và 9 bài viết/thông báo; đối chiếu sau chuẩn hóa có 13 Markdown, khoảng 246 nghìn ký tự và 633 chunks. Xác định các chủ đề dịch vụ sinh viên còn thiếu gồm ký túc xá, bảo hiểm y tế và thủ tục cấp bảng điểm/giấy xác nhận. | README.md, data/landing/, data/standardized/, PHAN_CONG.md | Done |   
| Kế hoạch bổ sung nguồn tin tức | Xác định quy trình thêm 3–5 bài từ ctt.hust.edu.vn: cập nhật danh sách URL, kiểm tra selector chỉ lấy nội dung chính, chuẩn hóa Markdown và reindex từ baseline 633 chunks. Chưa có dữ liệu mới hoặc commit ingestion trên main, vì vậy không ghi nhận là đã hoàn thành triển khai. | src/task2_crawl_news.py, src/task3_convert_markdown.py, src/task4_chunking_indexing.py | Partial |   
| Phân tích benchmark retrieval | Đối chiếu 5 cấu hình trên cùng 18 câu hỏi, top_k=5. Dense-only đạt RAGAS trung bình 0.806 và hit-rate 0.889; hybrid + RRF đạt 0.763 và 0.778; hybrid + cross-encoder đạt tốt nhất với 0.830 và 0.944. Xác nhận kết luận phải dựa trên số liệu, không mặc định hybrid luôn tốt hơn dense. | group_project/evaluation/RESULT.md, group_project/evaluation/eval_results.json | Done |   
| Phân tích lỗi và trade-off | Rà soát ba nhóm lỗi chính: tài liệu gần trùng làm BM25/RRF đẩy chunk đúng khỏi top-5; mệnh đề trả lời bị cắt qua biên chunk; câu hỏi hb-01 trượt ở mọi cấu hình. Đối chiếu lợi ích của cross-encoder với chi phí latency 28 ms → 557 ms/query. | Mục “Worst performers”, “Recommendations” và “Bonus experiments” trong group_project/evaluation/RESULT.md | Done |   
| Rà soát khả năng chạy lại | Kiểm tra tĩnh các bước cài đặt, biến môi trường, lệnh index/evaluate/test và điểm vào Streamlit. Việc dựng lại toàn bộ từ clone sạch chưa được thực hiện trong commit hiện có nên chỉ ghi nhận phần review tài liệu. | README.md, .env.example, pyproject.toml, tests/ | Partial |   
| Hoàn thiện báo cáo cá nhân | Loại bỏ nội dung mẫu, phân biệt rõ kết quả nhóm với phần tôi trực tiếp rà soát và không nhận ownership đối với code chưa có commit/PR của mình. | reports/2A202602479-DinhCongTu.md | Done |   
   
**Quyết định kỹ thuật quan trọng**  
1. **Quyết định:** Không chọn hybrid + RRF làm cấu hình mặc định chỉ vì nó kết hợp dense và lexical search.  
   
 **Lý do/evidence:** Trên cùng golden dataset 18 câu, hybrid + RRF giảm hit-rate từ 0.889 xuống 0.778 và giảm trung bình bốn metric RAGAS từ 0.806 xuống 0.763 so với dense-only. Phân tích lỗi cho thấy BM25 xếp cao các tài liệu gần trùng; RRF chỉ nhìn thứ hạng nên có thể đẩy chunk đúng ra khỏi top-5. Nếu ưu tiên chất lượng, cross-encoder cho hit-rate 0.944 và RAGAS trung bình 0.830.  
   
 **Trade-off:** Dense-only nhanh và đơn giản nhưng có thể bỏ sót truy vấn thiên về từ khóa. Cross-encoder cải thiện chất lượng nhưng làm retrieval tăng từ khoảng 28 ms lên 557 ms/query và phải nạp mô hình khoảng 2.2 GB.  
2. **Quyết định:** Khi bổ sung nguồn mới, chỉ index phần nội dung chính và đo lại trên baseline 633 chunks thay vì đưa toàn bộ HTML trang vào corpus.  
   
 **Lý do/evidence:** Tài liệu dự án ghi nhận crawl toàn trang có thể tạo khoảng 40 nghìn ký tự/bài, trong đó phần lớn là menu điều hướng lặp; selector nội dung đưa bài về khoảng 0.8–8.6 nghìn ký tự hữu ích. Nội dung lặp làm tăng nhiễu cho BM25 và có thể làm sai thứ hạng RRF.  
   
 **Trade-off:** Selector riêng cho từng kiểu trang cần bảo trì khi giao diện website thay đổi; đổi lại corpus sạch hơn, giảm chunk rác và citation dễ kiểm chứng hơn.  
**Kiểm thử và kết quả**  
- **Phạm vi tôi đã kiểm tra:** cấu trúc corpus, bảng kết quả 5 cấu hình, các trường hợp thất bại và sự nhất quán giữa kết luận với số liệu trong RESULT.md và eval_results.json.  
- **Kết quả đã đối chiếu:** baseline có 13 tài liệu/633 chunks; golden dataset có 18 câu; dense-only tốt hơn hybrid + RRF trong A/B chính; Config C (hybrid + cross-encoder) có chất lượng tổng thể tốt nhất nhưng latency cao hơn gần 20 lần.  
- **Bằng chứng kiểm thử chung của repository:**pytest -q được báo cáo 33/33 pass; python -m scripts.verify_golden_dataset được báo cáo 18/18 context xuất hiện trong corpus. Đây là kết quả chung đã lưu trong repository, không phải một lần chạy độc lập do tôi thực hiện.  
- **Lỗi/rủi ro phát hiện khi review:** tên file mô tả deliverable trong bản mẫu cũ không khớp repository thực tế; dự án dùng src/task5_semantic_search.py, src/task6_lexical_search.py, src/task7_reranking.py và src/task8_pageindex_vectorless.py. Báo cáo này sử dụng đúng đường dẫn hiện có để người chấm có thể kiểm tra.  
**Điều còn hạn chế**  
- Phần bổ sung 3–5 bài tin tức và lần dựng lại từ clone sạch chưa có artifact, commit hoặc số đo mới trên repository nhóm; vì vậy hai mục này được đánh dấu Partial, không ghi khống là Done.  
- Cỡ mẫu 18 câu còn nhỏ: chênh một trường hợp tương đương khoảng 0.056, nên các kết luận hiện tại phù hợp để chọn cấu hình cho dự án nhưng chưa đủ để khái quát rộng.  
- Hit-rate đang dựa trên khớp chuỗi expected_context; hai đoạn tương đương về nghĩa vẫn có thể bị tính là miss.  
- Nếu có thêm thời gian, thay đổi đầu tiên tôi sẽ thực hiện là hoàn thành ingestion 3–5 nguồn còn thiếu, ghi số ký tự từng bài và số chunks trước/sau, sau đó chạy lại A/B để kiểm tra dữ liệu mới có làm thay đổi kết luận dense so với hybrid hay không.  
**Xác nhận đóng góp**  
Tôi xác nhận nội dung trên phân biệt rõ phần tôi trực tiếp rà soát với kết quả chung của nhóm và có thể giải thích lại các số liệu, quyết định và giới hạn trong buổi demo.  
