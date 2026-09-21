"""
Task 1 — Thu thập tài liệu chính sách/quy định.

Đề tài nhóm: **Dịch vụ sinh viên Đại học Bách khoa Hà Nội (HUST)**.

Nguồn: cổng thông tin công khai ctt.hust.edu.vn (Ban Công tác sinh viên) —
văn bản quy chế/quyết định đã ban hành, không cần đăng nhập, không vượt WAF.

Chạy lại nhiều lần an toàn: file đã tải và còn hợp lệ sẽ được bỏ qua.
"""

from pathlib import Path

import requests


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)

# Tên file không dấu, phản ánh đúng nội dung văn bản.
# title: dùng làm nhãn citation trong câu trả lời nên phải nói rõ văn bản nào,
# không lấy dòng đầu PDF (mọi văn bản đều mở đầu bằng "BỘ GIÁO DỤC VÀ ĐÀO TẠO").
SOURCES: dict[str, dict[str, str]] = {
    "hust-quy-che-dao-tao-2023.pdf": {
        "title": "Quy chế đào tạo ĐH Bách khoa Hà Nội (QĐ 4600/QĐ-ĐHBK, 2023)",
        "url": (
            "https://ctt.hust.edu.vn/Upload/Nguyen%20Quoc%20Dat/files/DTDH_QDQC/"
            "Hoctap/QCDT-2023-upload.pdf"
        ),
    },
    "hust-quy-che-dao-tao-2025.pdf": {
        "title": "Quy chế đào tạo ĐH Bách khoa Hà Nội (QĐ 5445/QĐ-ĐHBK, 2025)",
        "url": (
            "https://ctt.hust.edu.vn/Upload/Nguy%E1%BB%85n%20Qu%E1%BB%91c%20%C4%90"
            "%E1%BA%A1t/files/DTDH_QDQC/Hoctap/QCDT_2025_5445_QD-DHBK.pdf"
        ),
    },
    "hust-quyet-dinh-hoc-phi-2025-2026.pdf": {
        "title": "Quyết định phê duyệt mức học phí năm học 2025-2026 (ĐHBK Hà Nội)",
        "url": (
            "https://ctt.hust.edu.vn/Upload/Nguy%E1%BB%85n%20Qu%E1%BB%91c%20%C4%90"
            "%E1%BA%A1t/files/DTDH_QDQC/Hocphi/2025-2026/"
            "QD%20HOC%20PHI%20-%202025-2026-final.pdf"
        ),
    },
    "hust-quy-che-dao-tao-tin-chi-en.pdf": {
        "title": "HUST Regulation on the Credit-based Training System (English)",
        "url": (
            "https://hust.edu.vn/uploads/sys/quality-assurance/2019/04/"
            "ee-1-2-4-hust-regulation-on-the-credit-based-training-system"
            ".399978.17776.pdf"
        ),
    },
}

MIN_SIZE_BYTES = 1024


def setup_directory() -> None:
    """Tạo thư mục lưu tài liệu gốc."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {DATA_DIR}")


def _is_already_downloaded(path: Path) -> bool:
    return path.exists() and path.stat().st_size > MIN_SIZE_BYTES


def download_documents() -> None:
    """Tải ít nhất 3 PDF/DOCX từ nguồn công khai."""
    setup_directory()

    for filename, source in SOURCES.items():
        url = source["url"]
        target = DATA_DIR / filename
        if _is_already_downloaded(target):
            print(f"Skip (exists): {filename}")
            continue

        try:
            response = requests.get(
                url, timeout=60, headers={"User-Agent": USER_AGENT}
            )
            response.raise_for_status()
        except requests.RequestException as error:
            print(f"Failed: {filename} — {error}")
            continue

        payload = response.content
        if len(payload) <= MIN_SIZE_BYTES or not payload.startswith(b"%PDF"):
            print(f"Failed: {filename} — response is not a valid PDF")
            continue

        target.write_bytes(payload)
        print(f"Saved: {target.name} ({len(payload):,} bytes)")

    downloaded = [
        path for path in DATA_DIR.glob("*.pdf") if _is_already_downloaded(path)
    ]
    print(f"Legal documents available: {len(downloaded)}")
    if len(downloaded) < 3:
        raise RuntimeError("Cần tối thiểu 3 tài liệu chính sách hợp lệ")


if __name__ == "__main__":
    download_documents()
