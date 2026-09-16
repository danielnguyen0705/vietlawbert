"""
html_to_md.py - Bộ chuyển đổi hàng loạt tài liệu HTML sang Markdown giữ nguyên cấu trúc.
Khử sạch thẻ DOM rác của giao diện CMS và bảo toàn thẻ bảng biểu, tiêu đề pháp luật.
"""

from __future__ import annotations

import os
import sys
import glob
import logging
import re
from pathlib import Path
from bs4 import BeautifulSoup
import html2text

from configs.paths import RAW_HTML_DIR, PROCESSED_DIR, get_log_path

logger = logging.getLogger("VietLawBERT_HTMLConverter")


class HTMLConverter:
    def __init__(self):
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_links = True
        self.h2t.ignore_images = True
        self.h2t.body_width = 0
        self.h2t.protect_links = False

    @staticmethod
    def clean_html(html_content: str) -> str:
        """Dọn dẹp các thành phần điều hướng giao diện Web, chỉ giữ lại khung văn bản chính."""
        if not html_content:
            return ""

        soup = BeautifulSoup(html_content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "button", "iframe"]):
            tag.decompose()

        main_content = (
            soup.find("div", class_="fulltext")
            or soup.find("div", id="toanvancontent")
            or soup.find("div", class_="content")
            or soup.find("body")
        )
        return str(main_content) if main_content else html_content

    @staticmethod
    def normalize_legal_markdown(text: str) -> str:
        """Chuẩn hóa tiêu đề phân cấp tuân thủ thể thức văn bản pháp luật Việt Nam."""
        if not text:
            return ""

        patterns = [
            (r"^(?:#|\*)*\s*(Phần\s+[IVXLCDM\d]+)\s*[:\.\*]*\s*(.*)$", r"# \1\n\2"),
            (r"^(?:#|\*)*\s*(Chương\s+[IVXLCDM\d]+)\s*[:\.\*]*\s*(.*)$", r"## \1\n\2"),
            (r"^(?:#|\*)*\s*(Mục\s+\d+)\s*[:\.\*]*\s*(.*)$", r"### \1\n\2"),
            (r"^(?:#|\*)*\s*(Tiểu\s+mục\s+\d+)\s*[:\.\*]*\s*(.*)$", r"#### \1\n\2"),
            (r"^(?:#|\*)*\s*(Điều\s+\d+[a-zA-Z]*)\s*[:\.\*]*\s*(.*)$", r"##### \1\n\2"),
            (r"^(?:#|\*)*\s*(Phụ\s+lục\s*[IVXLCDM\d]*)\s*[:\.\*]*\s*(.*)$", r"# \1\n\2"),
        ]

        for pat, repl in patterns:
            text = re.sub(pat, repl, text, flags=re.MULTILINE | re.IGNORECASE)

        # Rút gọn các dòng trống liên tiếp
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def convert_file(self, html_path: Path, output_path: Path) -> bool:
        """Chuyển đổi một tệp HTML đơn lẻ sang Markdown chuẩn."""
        try:
            with open(html_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            cleaned = self.clean_html(content)
            raw_md = self.h2t.handle(cleaned)
            final_md = self.normalize_legal_markdown(raw_md)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8", newline="\n") as f_out:
                f_out.write(final_md)

            return True
        except Exception as exc:
            logger.error(f"Lỗi chuyển đổi tệp {html_path.name}: {exc}")
            return False

    def convert_all(self, input_dir: Path = RAW_HTML_DIR, output_dir: Path = PROCESSED_DIR) -> int:
        """Chuyển đổi hàng loạt toàn bộ thư mục HTML sang Markdown."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        html_files = sorted(list(input_dir.glob("*.html")))
        total = len(html_files)
        if total == 0:
            logger.warning(f"Không tìm thấy tệp HTML nào tại: {input_dir}")
            return 0

        logger.info(f"Bắt đầu chuyển đổi {total} tệp HTML sang Markdown...")
        success_count = 0

        for idx, file_path in enumerate(html_files, 1):
            out_file = output_dir / f"{file_path.stem}.md"
            if self.convert_file(file_path, out_file):
                success_count += 1

            if idx % 100 == 0 or idx == total:
                logger.info(f"Tiến độ chuyển đổi: {idx}/{total} tệp hoàn tất.")

        logger.info(f"✓ Hoàn tất: Chuyển đổi thành công {success_count}/{total} tệp Markdown.")
        return success_count


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | [%(levelname)s] | %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(get_log_path("html_to_md"), encoding="utf-8", mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    converter = HTMLConverter()
    converter.convert_all()


if __name__ == "__main__":
    main()