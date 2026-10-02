"""
fetch_zalo_data.py - Module tự động tải, đọc và thẩm định dữ liệu Zalo AI Challenge.
Kéo trực tiếp từ Hugging Face Hub (GreenNode/zalo-ai-legal-text-retrieval-vn).
Triệt tiêu lỗi file rỗng 0 bytes do Git LFS và cung cấp hàm nạp chuẩn hóa cho VietLawBench.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

from configs.paths import BENCHMARK_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | [%(levelname)s] | %(message)s")
logger = logging.getLogger("VietLawBERT_ZaloFetcher")


class ZaloDataFetcher:
    def __init__(self, target_dir: Path | str = BENCHMARK_DIR / "data"):
        self.target_dir = Path(target_dir)
        self.target_dir.mkdir(parents=True, exist_ok=True)
        self.raw_queries_file = self.target_dir / "zalo_queries.jsonl"
        self.standardized_file = self.target_dir / "zalo_3k2_clean.jsonl"
        self.repo_id = "GreenNode/zalo-ai-legal-text-retrieval-vn"

    def download_raw_dataset(self) -> Path:
        """
        Kéo tệp queries.jsonl trực tiếp từ Hugging Face qua hf_hub_download.
        Cơ chế này tự động xử lý Git LFS pointers, không bao giờ bị file rỗng 0 bytes.
        """
        if self.raw_queries_file.exists() and self.raw_queries_file.stat().st_size > 1000:
            logger.info("✓ Tệp Zalo queries đã tồn tại tại: %s (%d bytes). Bỏ qua bước tải.", 
                        self.raw_queries_file, self.raw_queries_file.stat().st_size)
            return self.raw_queries_file

        logger.info("Bắt đầu kéo dữ liệu câu hỏi Zalo AI từ Hugging Face: %s...", self.repo_id)
        try:
            from huggingface_hub import hf_hub_download
            downloaded_path = hf_hub_download(
                repo_id=self.repo_id,
                filename="queries.jsonl",
                repo_type="dataset",
                local_dir=str(self.target_dir),
                local_dir_use_symlinks=False
            )
            downloaded = Path(downloaded_path)
            if downloaded != self.raw_queries_file and downloaded.exists():
                downloaded.rename(self.raw_queries_file)

            logger.info("✓ Đã kéo tệp thành công về máy: %s (%d bytes)", 
                        self.raw_queries_file, self.raw_queries_file.stat().st_size)
            return self.raw_queries_file

        except Exception as e:
            logger.warning("Không thể dùng huggingface_hub (%s). Sử dụng urllib streaming dự phòng...", e)
            import urllib.request
            url = f"https://huggingface.co/datasets/{self.repo_id}/raw/main/queries.jsonl"
            urllib.request.urlretrieve(url, str(self.raw_queries_file))
            logger.info("✓ Đã tải qua đường dẫn dự phòng: %s", self.raw_queries_file)
            return self.raw_queries_file

    def read_and_parse_zalo_queries(self) -> List[Dict[str, Any]]:
        """
        Đọc tệp vừa tải về máy, bóc tách và chuẩn hóa danh sách câu hỏi.
        """
        raw_file = self.download_raw_dataset()
        logger.info("Đang đọc và giải mã cấu trúc dữ liệu từ: %s...", raw_file)

        parsed_records: List[Dict[str, Any]] = []

        with open(raw_file, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                clean_line = line.strip()
                if not clean_line:
                    continue
                try:
                    data = json.loads(clean_line)
                    # Cấu trúc tệp của Zalo AI: {"_id": "...", "text": "...", "metadata": ...}
                    q_id = str(data.get("_id") or data.get("question_id") or f"zalo_{line_idx:04d}")
                    q_text = str(data.get("text") or data.get("question") or data.get("query") or "").strip()

                    # Lọc bỏ các câu quá ngắn hoặc rỗng
                    if len(q_text) >= 15:
                        parsed_records.append({
                            "question_id": q_id,
                            "question": q_text,
                            "source": "Zalo_AI_Challenge_2021"
                        })
                except json.JSONDecodeError:
                    continue

        logger.info("✓ Đã nạp thành công %d câu hỏi thực tế của Zalo AI vào bộ nhớ.", len(parsed_records))

        # Lưu lại tệp JSONL sạch chuẩn hóa trong thư mục benchmark/data
        with open(self.standardized_file, "w", encoding="utf-8") as out_f:
            for item in parsed_records:
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

        return parsed_records

    def inspect_sample_questions(self, sample_size: int = 5):
        """
        Nghiệm thu tại chỗ: In mẫu 5 câu hỏi ra màn hình để kiểm tra văn phong đời thường.
        """
        records = self.read_and_parse_zalo_queries()
        total = len(records)

        print("\n" + "=" * 80)
        print(f"BÁO CÁO NGHIỆM THU DỮ LIỆU CÂU HỎI ZALO AI CHALLENGE (TỔNG SỐ: {total:,} CÂU)")
        print("=" * 80)

        for i, sample in enumerate(records[:sample_size], 1):
            print(f"[{i:02d}] ID: {sample['question_id']}")
            print(f"     Nội dung câu hỏi: \"{sample['question']}\"")
            print("-" * 80)

        print(f"✓ Tệp sạch đã lưu sẵn tại: {self.standardized_file.resolve()}")
        print("=" * 80 + "\n")
        return records


def get_zalo_benchmark_samples(limit: int = 500) -> List[Dict[str, Any]]:
    """
    Hàm cung cấp API cho build_vietlawbench.py gọi sang.
    """
    fetcher = ZaloDataFetcher()
    all_questions = fetcher.read_and_parse_zalo_queries()
    return all_questions[:limit]


def main():
    fetcher = ZaloDataFetcher()
    fetcher.inspect_sample_questions(sample_size=5)


if __name__ == "__main__":
    main()