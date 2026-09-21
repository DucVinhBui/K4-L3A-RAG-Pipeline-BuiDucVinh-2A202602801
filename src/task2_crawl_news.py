"""
Task 2 — Crawl bài viết/thông báo về dịch vụ sinh viên HUST.

Nguồn công khai: hust.edu.vn và ctt.hust.edu.vn (Cổng thông tin sinh viên).

Bài học rút ra khi làm task này: crawl cả trang thì ~95% nội dung là menu
điều hướng lặp trên mọi URL, làm nhiễu index và kéo precision xuống. Vì vậy
mỗi nguồn khai báo `selector` trỏ vào đúng khối nội dung bài viết.

Chiến lược: Crawl4AI (headless chromium) là đường chính; nếu browser chưa cài
hoặc trang lỗi thì fallback sang requests + MarkItDown để pipeline không đứng.

Cài browser trước khi chạy:
    python -m playwright install chromium
"""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)

MIN_CONTENT_CHARS = 300

# selector: khối chứa nội dung bài viết; title: ghi đè khi <title> của trang
# là tên portal chung chung thay vì tiêu đề bài.
ARTICLES: list[dict] = [
    {
        "url": "https://ctt.hust.edu.vn/DisplayWeb/DisplayBaiViet?baiviet=43466",
        "selector": "div.col-md-9",
        "title": "Thông báo về mức học phí đối với các chương trình đào tạo năm học 2024-2025",
        "slug": "thong-bao-muc-hoc-phi-2024-2025",
    },
    {
        "url": "https://ctt.hust.edu.vn/DisplayWeb/DisplayBaiViet?baiviet=43473",
        "selector": "div.col-md-9",
        "title": "Kết quả xét cấp học bổng khuyến khích học tập",
        "slug": "ket-qua-xet-hoc-bong-khuyen-khich-hoc-tap",
    },
    {
        "url": "https://hust.edu.vn/vi/sinh-vien/hoc-bong-hoc-phi/hoc-bong-645445.html",
        "selector": "div.bodytext",
        "title": "Học bổng dành cho sinh viên Đại học Bách khoa Hà Nội",
        "slug": "hoc-bong-tong-quan",
    },
    {
        "url": "https://hust.edu.vn/vi/sinh-vien/ho-tro-sinh-vien/bieu-do-ke-hoach-hoc-tap-654590.html",
        "selector": "div.bodytext",
        "title": "Biểu đồ kế hoạch học tập của sinh viên",
        "slug": "bieu-do-ke-hoach-hoc-tap",
    },
    {
        "url": "https://hust.edu.vn/vi/news/tin-tuc-su-kien/hoc-bong-bach-khoa-cho-tan-sinh-vien-k70-chinh-phuc-ngay-tu-nam-hoc-dau-tien-655566.html",
        "selector": "div.bodytext",
        "title": "Học bổng Bách khoa chờ tân sinh viên K70 chinh phục ngay từ năm học đầu tiên",
        "slug": "hoc-bong-tan-sinh-vien-k70",
    },
    {
        "url": "https://hust.edu.vn/vi/news/tin-tuc-su-kien/bach-khoa-ha-noi-danh-khoang-70-ty-dong-lam-quy-hoc-bong-khuyen-khich-hoc-tap-655145.html",
        "selector": "div.bodytext",
        "title": "Bách khoa Hà Nội dành khoảng 70 tỷ đồng làm Quỹ học bổng Khuyến khích học tập",
        "slug": "quy-hoc-bong-khuyen-khich-hoc-tap-70-ty",
    },
    {
        "url": "https://hust.edu.vn/vi/sinh-vien/cong-tac-sinh-vien/tan-sinh-vien-k71-huong-dan-di-chuyen-va-gui-xe-tai-truong-654629.html",
        "selector": "div.bodytext",
        "title": "Tân sinh viên K71: hướng dẫn di chuyển và gửi xe tại trường",
        "slug": "huong-dan-di-chuyen-va-gui-xe",
    },
    {
        "url": "https://hust.edu.vn/vi/sinh-vien/cong-tac-sinh-vien/ke-hoach-le-tot-nghiep-dot-thang-9-2026-654630.html",
        "selector": "div.bodytext",
        "title": "Kế hoạch Lễ tốt nghiệp đợt tháng 9/2026",
        "slug": "ke-hoach-le-tot-nghiep-thang-9-2026",
    },
    {
        "url": "https://hust.edu.vn/vi/sinh-vien/ho-tro-sinh-vien/dai-hoc-bach-khoa-ha-noi-dong-hanh-cung-sinh-vien-thao-go-ap-luc-dinh-huong-tuong-lai-654627.html",
        "selector": "div.bodytext",
        "title": "Đại học Bách khoa Hà Nội đồng hành cùng sinh viên tháo gỡ áp lực, định hướng tương lai",
        "slug": "ho-tro-tam-ly-dinh-huong-sinh-vien",
    },
]

ARTICLE_URLS = [article["url"] for article in ARTICLES]


def _clean_markdown(text: str) -> str:
    """Bỏ dòng trống thừa và khoảng trắng cuối dòng."""
    lines = [line.rstrip() for line in text.splitlines()]
    kept: list[str] = []
    for line in lines:
        if kept and not line.strip() and not kept[-1].strip():
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def _first_heading(markdown: str) -> str | None:
    match = re.search(r"^#{1,6}\s+(.+)$", markdown, re.MULTILINE)
    return match.group(1).strip() if match else None


def _fetch_with_requests(url: str, selector: str) -> str:
    """Fallback tĩnh: tải HTML, cắt theo selector đơn giản rồi convert Markdown."""
    import io

    import requests
    from markitdown import MarkItDown

    response = requests.get(url, timeout=60, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    html = response.text

    class_name = selector.split(".", 1)[-1]
    start = re.search(rf'<div[^>]*class="[^"]*{re.escape(class_name)}[^"]*"', html)
    if start:
        html = html[start.start():]

    converted = MarkItDown().convert_stream(
        io.BytesIO(html.encode("utf-8")), file_extension=".html"
    )
    return _clean_markdown(converted.text_content)


async def crawl_article(url: str) -> dict:
    """Crawl một URL, trả về dict có url/title/date_crawled/content_markdown."""
    config = next(
        (item for item in ARTICLES if item["url"] == url),
        {"url": url, "selector": "body"},
    )
    selector = config["selector"]
    markdown = ""
    page_title = ""

    try:
        from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

        run_config = CrawlerRunConfig(css_selector=selector, word_count_threshold=3)
        async with AsyncWebCrawler(verbose=False) as crawler:
            result = await crawler.arun(url=url, config=run_config)
            markdown = _clean_markdown(str(result.markdown or ""))
            page_title = (result.metadata or {}).get("title", "") or ""
    except Exception as error:  # browser thiếu, timeout, WAF...
        print(f"  crawl4ai unavailable ({error}); falling back to requests")

    if len(markdown) < MIN_CONTENT_CHARS:
        markdown = await asyncio.to_thread(_fetch_with_requests, url, selector)

    if len(markdown) < MIN_CONTENT_CHARS:
        raise ValueError(f"nội dung thu được quá ngắn ({len(markdown)} ký tự)")

    title = (
        config.get("title")
        or _first_heading(markdown)
        or re.sub(r"\s+", " ", page_title).strip()
        or url
    )
    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(timespec="seconds"),
        "content_markdown": markdown,
    }


async def crawl_all() -> None:
    """Crawl và lưu từng bài thành một file JSON."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    saved = 0
    for index, config in enumerate(ARTICLES, 1):
        output = DATA_DIR / f"article_{index:02d}_{config['slug']}.json"
        try:
            article = await crawl_article(config["url"])
            output.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            saved += 1
            print(
                f"Saved: {output.name} "
                f"({len(article['content_markdown']):,} chars)"
            )
        except Exception as error:
            print(f"Failed: {config['url']} — {error}")

    print(f"News articles saved: {saved}")
    if saved < 5:
        raise RuntimeError("Cần tối thiểu 5 bài viết hợp lệ")


if __name__ == "__main__":
    asyncio.run(crawl_all())
