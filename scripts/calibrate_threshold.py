"""Hiệu chỉnh SCORE_THRESHOLD cho fallback ở Task 9.

Threshold so sánh với cosine score gốc của dense retrieval, nên phải đo trên
chính corpus của nhóm: không có con số đúng cho mọi corpus.

Cách đo: chạy semantic_search cho hai tập query, lấy best cosine score của
từng query, rồi chọn ngưỡng tách hai phân phối tốt nhất (max accuracy, ưu
tiên ngưỡng nằm giữa khoảng trống giữa hai nhóm).

    python -m scripts.calibrate_threshold
"""

from src.task5_semantic_search import semantic_search


IN_DOMAIN = [
    "Học bổng khuyến khích học tập có mấy mức?",
    "Sinh viên bị cảnh báo học tập trong trường hợp nào?",
    "Mức học phí năm học 2025-2026 được quy định thế nào?",
    "Điều kiện để được xét tốt nghiệp đại học là gì?",
    "Sinh viên được rút học phần trong thời gian nào?",
    "Học bổng Trần Đại Nghĩa dành cho đối tượng nào?",
    "Thang điểm đánh giá kết quả học tập được quy định ra sao?",
    "Sinh viên gửi xe ở đâu trong trường?",
    "Lễ tốt nghiệp đợt tháng 9/2026 diễn ra khi nào?",
    "Buộc thôi học áp dụng với sinh viên nào?",
]

OUT_OF_DOMAIN = [
    "Giá vé tàu Hà Nội đi Sài Gòn là bao nhiêu?",
    "Cách nấu phở bò truyền thống?",
    "Tỷ giá USD hôm nay thế nào?",
    "Đội tuyển Việt Nam thi đấu lúc mấy giờ?",
    "Triệu chứng của bệnh cúm A là gì?",
    "Cách cài đặt Docker trên Ubuntu?",
    "Thời tiết Đà Nẵng cuối tuần này ra sao?",
    "Lãi suất vay mua nhà của ngân hàng nào thấp nhất?",
]


def best_score(query: str) -> float:
    results = semantic_search(query, top_k=5)
    return results[0]["score"] if results else 0.0


def main() -> None:
    in_scores = [(q, best_score(q)) for q in IN_DOMAIN]
    out_scores = [(q, best_score(q)) for q in OUT_OF_DOMAIN]

    print("IN-DOMAIN (best cosine)")
    for query, score in sorted(in_scores, key=lambda item: item[1]):
        print(f"  {score:.4f}  {query}")
    print("\nOUT-OF-DOMAIN (best cosine)")
    for query, score in sorted(out_scores, key=lambda item: item[1], reverse=True):
        print(f"  {score:.4f}  {query}")

    positives = [score for _, score in in_scores]
    negatives = [score for _, score in out_scores]
    total = len(positives) + len(negatives)

    best = (0.0, -1.0)  # (threshold, accuracy)
    candidate = 0.0
    while candidate <= 1.0001:
        correct = sum(score >= candidate for score in positives) + sum(
            score < candidate for score in negatives
        )
        accuracy = correct / total
        if accuracy > best[1]:
            best = (candidate, accuracy)
        candidate += 0.01

    print(
        f"\nin-domain  min={min(positives):.4f} "
        f"mean={sum(positives)/len(positives):.4f} max={max(positives):.4f}"
    )
    print(
        f"out-domain min={min(negatives):.4f} "
        f"mean={sum(negatives)/len(negatives):.4f} max={max(negatives):.4f}"
    )
    gap_low, gap_high = max(negatives), min(positives)
    if gap_high > gap_low:
        print(f"Khoảng tách sạch: ({gap_low:.4f}, {gap_high:.4f})")
        print(f"Đề xuất SCORE_THRESHOLD = {(gap_low + gap_high) / 2:.2f}")
    else:
        print("Hai phân phối chồng lấn — không có ngưỡng tách sạch.")
        print(f"Ngưỡng accuracy cao nhất = {best[0]:.2f} (accuracy {best[1]:.2%})")


if __name__ == "__main__":
    main()
