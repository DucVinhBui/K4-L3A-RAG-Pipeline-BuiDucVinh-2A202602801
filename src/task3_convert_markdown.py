"""
Task 3 — Chuẩn hóa dữ liệu sang Markdown.

Mọi file trong data/standardized/ có cùng một front matter YAML để Task 4 đọc
metadata mà không cần đoán từ tên file:

    ---
    title: ...
    source: ...
    doc_type: legal | news
    url: ...            # "null" nếu không có
    date_crawled: ...   # chỉ có với news
    ---

PDF/DOCX được convert bằng MarkItDown; JSON news đã có sẵn Markdown từ Task 2.
Chạy lại an toàn: file .md được ghi đè theo tên gốc nên không sinh bản trùng.
"""

import json
import re
from pathlib import Path

from .task1_collect_legal_docs import SOURCES as LEGAL_SOURCES


LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"

MIN_OUTPUT_CHARS = 200


def _escape(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def build_front_matter(
    *,
    title: str,
    source: str,
    doc_type: str,
    url: str | None,
    date_crawled: str | None = None,
) -> str:
    lines = [
        "---",
        f'title: "{_escape(title)}"',
        f'source: "{_escape(source)}"',
        f"doc_type: {doc_type}",
        f'url: {"null" if not url else chr(34) + _escape(url) + chr(34)}',
    ]
    if date_crawled:
        lines.append(f'date_crawled: "{_escape(date_crawled)}"')
    lines.append("---")
    return "\n".join(lines) + "\n\n"


def _tidy(text: str) -> str:
    """Gộp dòng trống thừa và bỏ ký tự điều khiển từ PDF."""
    text = text.replace("\x00", "").replace("\r\n", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _title_from_markdown(text: str, fallback: str) -> str:
    match = re.search(r"^#{1,6}\s+(.+)$", text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    for line in text.splitlines():
        if len(line.strip()) > 10:
            return line.strip()[:120]
    return fallback


def convert_legal_docs() -> int:
    """Convert PDF/DOCX trong landing/legal sang standardized/legal."""
    from markitdown import MarkItDown

    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)
    converter = MarkItDown()

    written = 0
    for path in sorted(legal_dir.iterdir()):
        if path.suffix.lower() not in {".pdf", ".doc", ".docx"}:
            continue
        try:
            body = _tidy(converter.convert(str(path)).text_content)
        except Exception as error:
            print(f"Failed: {path.name} — {error}")
            continue

        if len(body) < MIN_OUTPUT_CHARS:
            print(f"Skip (empty after convert): {path.name}")
            continue

        source_info = LEGAL_SOURCES.get(path.name, {})
        header = build_front_matter(
            title=source_info.get("title")
            or _title_from_markdown(body, path.stem),
            source=path.name,
            doc_type="legal",
            url=source_info.get("url"),
        )
        (output_dir / f"{path.stem}.md").write_text(
            header + body, encoding="utf-8"
        )
        written += 1
        print(f"Converted legal: {path.stem}.md ({len(body):,} chars)")
    return written


def convert_news_articles() -> int:
    """Convert JSON trong landing/news sang standardized/news."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for path in sorted(news_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        body = _tidy(data.get("content_markdown", ""))
        if len(body) < MIN_OUTPUT_CHARS:
            print(f"Skip (too short): {path.name}")
            continue

        header = build_front_matter(
            title=data["title"],
            source=path.name,
            doc_type="news",
            url=data.get("url"),
            date_crawled=data.get("date_crawled"),
        )
        (output_dir / f"{path.stem}.md").write_text(
            header + body, encoding="utf-8"
        )
        written += 1
        print(f"Converted news: {path.stem}.md ({len(body):,} chars)")
    return written


def convert_all() -> None:
    """Convert toàn bộ dữ liệu landing."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    legal = convert_legal_docs()
    news = convert_news_articles()
    print(f"Saved Markdown to: {OUTPUT_DIR} (legal={legal}, news={news})")
    if legal < 3 or news < 5:
        raise RuntimeError("Cần tối thiểu 3 legal và 5 news sau chuẩn hóa")


if __name__ == "__main__":
    convert_all()
