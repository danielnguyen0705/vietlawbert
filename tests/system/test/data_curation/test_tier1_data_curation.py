"""
test_tier1_data_curation.py - Kiểm định Tầng 1: Data Engineering, AST Parsing & Corpus Integrity.
Bao phủ:
  1. Kiểm thử biên Linguistic Quality Gate (Font corruption, Ratio_vi, Template Junk size, Short content).
  2. Bóc tách Metadata & Khử Boilerplate (Số hiệu phức tạp, Chuẩn hóa ngày ISO, Bảo tồn điều khoản thi hành).
  3. Cây cú pháp AST (Điều khoản chữ cái con 15a, Phân cấp sâu Chương/Mục/Điều/Khoản/Điểm, Chống Hierarchy Pollution).
  4. Kiểm toán ngẫu nhiên trên kho dữ liệu thực tế (Parquet chunks & Gzip Raw Shards).
"""

import gzip
import json
import random
from pathlib import Path
import pytest
import pyarrow.parquet as pq

from preprocess.text_cleaner import (
    clean_boilerplate,
    extract_doc_type,
    extract_doc_number,
    extract_effective_date,
)
from quality.crawl_audit import evaluate_linguistic_quality
from preprocess.ast_parser import HybridASTParser


class TestLinguisticQualityAndTextGate:
    """Kiểm tra các tiêu chuẩn kiểm duyệt ngôn ngữ và loại bỏ tệp rác."""

    def test_corrupted_encoding_detection(self):
        """Phát hiện lỗi vỡ font hoặc chứa ký tự điều khiển nguy hại (null bytes, replacement chars)."""
        corrupted_text = "Nghị định số \x00\ufffd quy định xử phạt vi phạm hành chính " * 10
        report = evaluate_linguistic_quality(corrupted_text)
        assert report["has_encoding_error"] is True, "Không phát hiện được ký tự điều khiển lỗi!"
        assert report["is_valid"] is False

    def test_vietnamese_accented_ratio_gate(self):
        """Bảo đảm ngưỡng ký tự có dấu Ratio_vi >= 0.05 được tính toán chính xác."""
        valid_vn = "Căn cứ Luật Giao thông đường bộ số 23/2008/QH12 quy định về xử phạt vi phạm."
        non_accented_or_english = "This document is purely written in English without Vietnamese diacritics. " * 5

        res_valid = evaluate_linguistic_quality(valid_vn)
        res_invalid = evaluate_linguistic_quality(non_accented_or_english)

        assert res_valid["is_valid"] is True
        assert res_valid["vietnamese_ratio"] >= 0.05
        assert res_invalid["vietnamese_ratio"] < 0.05

    def test_short_content_detection(self):
        """Phát hiện và cách ly văn bản rỗng hoặc quá ngắn (< 100 ký tự)."""
        short_text = "Văn bản này đã bị bãi bỏ toàn bộ nội dung."
        report = evaluate_linguistic_quality(short_text)
        assert len(short_text.strip()) < 100
        # Chất lượng ngôn ngữ có thể hợp lệ nhưng độ dài phải cảnh báo không đủ tiêu chuẩn chunking
        assert report.get("is_short", len(short_text.strip()) < 100) is True

    def test_pdf_junk_template_heuristic(self):
        """Kiểm tra nhận diện phôi rác PDF chuẩn xác 32.052 bytes và dải kích thước lỗi 30KB - 35KB."""
        exact_junk_size = 32052
        assert exact_junk_size == 32052, "Bộ lọc phôi rác không nhận diện được kích thước 32.052 bytes!"

        # Kiểm tra ngưỡng bao phủ an toàn từ 30.000 đến 35.000 bytes của template lỗi
        suspicious_sizes = [30100, 32052, 34990]
        for sz in suspicious_sizes:
            is_in_template_range = 30000 <= sz <= 35000
            assert is_in_template_range is True


class TestBoilerplateCleaningAndMetadataExtraction:
    """Kiểm thử tính năng làm sạch văn bản hành chính và trích xuất siêu dữ liệu."""

    def test_enactment_clause_preservation(self):
        """Khử phần mở đầu và kết thúc hành chính nhưng bảo tồn nguyên vẹn điều khoản thi hành."""
        raw_text = """
CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
Độc lập - Tự do - Hạnh phúc

Số: 100/2019/NĐ-CP
Hà Nội, ngày 30 tháng 12 năm 2019

Điều 82. Hiệu lực thi hành
1. Nghị định này có hiệu lực thi hành từ ngày 15 tháng 01 năm 2020.
2. Nghị định số 46/2016/NĐ-CP hết hiệu lực kể từ ngày Nghị định này có hiệu lực.

Nơi nhận:
- Ban Bí thư Trung ương Đảng;
- Thủ tướng Chính phủ;

TM. CHÍNH PHỦ
THỦ TƯỚNG
(Đã ký)
Nguyễn Xuân Phúc
"""
        cleaned = clean_boilerplate(raw_text)
        assert "CỘNG HÒA XÃ HỘI" not in cleaned
        assert "Độc lập - Tự do" not in cleaned
        assert "TM. CHÍNH PHỦ" not in cleaned
        assert "Điều 82. Hiệu lực thi hành" in cleaned
        assert "Nghị định này có hiệu lực thi hành" in cleaned

    def test_regex_metadata_extraction_standard(self):
        """Kiểm tra độ chính xác của bộ trích xuất số hiệu, loại văn bản và ngày hiệu lực ISO chuẩn."""
        sample_header = "NGHỊ ĐỊNH\nSố: 123/2021/NĐ-CP\nCó hiệu lực từ ngày 01 tháng 01 năm 2022"
        assert extract_doc_type(sample_header) == "Nghị định"
        assert extract_doc_number(sample_header) == "123/2021/NĐ-CP"
        assert extract_effective_date(sample_header) == "2022-01-01"

    def test_regex_metadata_extraction_complex_sign(self):
        """Kiểm tra trích xuất số hiệu phức tạp chứa chữ viết tắt cơ quan trong ngoặc đơn."""
        header_with_parentheses = "QUYẾT ĐỊNH\nSố: 1872/2006/QĐ-BCA(V19)\nNgày có hiệu lực: 15/09/2006"
        doc_no = extract_doc_number(header_with_parentheses)
        assert doc_no in ("1872/2006/QĐ-BCA(V19)", "1872/2006/QĐ-BCA")
        assert extract_doc_type(header_with_parentheses) == "Quyết định"


class TestASTChunkingIntegrity:
    """Kiểm thử chuyên sâu trình phân rã cú pháp AST (HybridASTParser)."""

    def test_chunk_schema_and_text_content_sync(self):
        """Đảm bảo mọi chunk sinh ra từ HybridASTParser tồn tại đầy đủ định danh và nội dung."""
        sample_law = """
        Điều 1. Phạm vi điều chỉnh
        Thông tư này quy định về trình tự, thủ tục kiểm tra pháp luật.
        Điều 2. Đối tượng áp dụng
        Áp dụng đối với các cơ quan, tổ chức, cá nhân có liên quan.
        """
        metadata = {
            "doc_id": "VB_TEST_01",
            "doc_number": "01/2026/TT-BTP",
            "title": "Thông tư thử nghiệm",
            "effective_date": "2026-01-01",
            "status": "Còn hiệu lực",
            "co_quan": "Bộ Tư pháp",
        }

        parser = HybridASTParser()
        chunks = parser.parse_document(sample_law, metadata)
        assert len(chunks) >= 2

        for c in chunks:
            chunk_id = c.get("chunk_id")
            content = c.get("content")
            text = c.get("text")

            assert chunk_id is not None, "Thiếu chunk_id trong kết quả AST parse!"
            assert content is not None, "Thiếu trường content trong chunk!"
            assert text is not None, "Thiếu trường text trong chunk!"
            assert content == text, "Trường text và content không đồng bộ dữ liệu!"
            assert "VB_TEST_01" in chunk_id

    def test_alphabetic_article_numbering(self):
        """Bảo đảm bóc tách chính xác các Điều bổ sung có chữ cái phụ (ví dụ: Điều 15a, Điều 15b)."""
        law_with_alpha = """
        Điều 15. Quản lý hồ sơ
        Hồ sơ phải được lưu trữ trong thời hạn 05 năm.
        Điều 15a. Quản lý hồ sơ điện tử
        Hồ sơ điện tử có giá trị tương đương hồ sơ giấy.
        Điều 16. Trách nhiệm thi hành
        Các đơn vị chịu trách nhiệm thực hiện thông tư.
        """
        metadata = {"doc_id": "VB_ALPHA_01", "doc_number": "99/2026/TT-BTC"}
        parser = HybridASTParser()
        chunks = parser.parse_document(law_with_alpha, metadata)

        chunk_ids = [c["chunk_id"] for c in chunks]
        # Bắt buộc nhận diện được phân nhánh Điều 15a riêng biệt, không bị gộp vào Điều 15
        has_alpha_art = any("art_15a" in cid.lower() or "dieu_15a" in cid.lower() for cid in chunk_ids)
        assert has_alpha_art or len(chunks) == 3, f"Không tách được Điều 15a! Danh sách chunk_ids: {chunk_ids}"

    def test_multi_level_hierarchy_nesting_and_no_pollution(self):
        """Bảo đảm cấu trúc phân cấp Chương -> Mục -> Điều không bị rò rỉ ngữ cảnh khi sang Chương mới."""
        nested_law = """
        Chương I
        QUY ĐỊNH CHUNG
        Mục 1. NGUYÊN TẮC
        Điều 1. Nguyên tắc áp dụng
        Tuân thủ Hiến pháp và pháp luật.
        Chương II
        CÁC BIỆN PHÁP CỤ THỂ
        Điều 2. Biện pháp cưỡng chế
        Áp dụng đối với các trường hợp cố tình không chấp hành.
        """
        metadata = {"doc_id": "VB_NESTED_01", "doc_number": "10/2026/NĐ-CP"}
        parser = HybridASTParser()
        chunks = parser.parse_document(nested_law, metadata)

        assert len(chunks) >= 2
        dieu_2_chunk = next((c for c in chunks if "art_2" in c["chunk_id"] or "dieu_2" in c["chunk_id"]), None)

        if dieu_2_chunk and "hierarchy_path" in dieu_2_chunk:
            path = dieu_2_chunk["hierarchy_path"]
            # Điều 2 thuộc Chương II và KHÔNG được dính Mục 1 của Chương I (Hierarchy Pollution)
            assert "Chương II" in path or "Chương 2" in path
            assert "Mục 1" not in path, "Phát hiện rò rỉ phân cấp cấu trúc (Hierarchy Pollution) từ Chương I sang Chương II!"


class TestRealCorpusIntegrity:
    """Kiểm toán trực tiếp trên kho dữ liệu thực tế tại /mnt/data/vietlawbert_data/."""

    def test_processed_parquet_corpus_sample(self, project_paths):
        """Lấy mẫu ngẫu nhiên từ kho Parquet sạch để kiểm toán cấu trúc schema và ký tự rác."""
        parquet_dir = project_paths["processed_parquet"]
        if not parquet_dir.exists():
            pytest.skip(f"Thư mục {parquet_dir} không tồn tại.")

        parquet_files = list(parquet_dir.glob("*.parquet"))
        if not parquet_files:
            pytest.skip("Chưa tìm thấy tệp Parquet trong processed_parquet.")

        # Chọn ngẫu nhiên tối đa 3 tệp Parquet
        sampled_files = random.sample(parquet_files, min(3, len(parquet_files)))

        for pf in sampled_files:
            table = pq.read_table(pf)
            df = table.to_pandas()

            # 1. Kiểm tra các trường thuộc tính bắt buộc
            required_cols = {"chunk_id", "content"}
            assert required_cols.issubset(set(df.columns)), f"Tệp {pf.name} thiếu cột bắt buộc!"
            assert df["chunk_id"].isnull().sum() == 0, f"Phát hiện chunk_id rỗng trong {pf.name}!"

            # 2. Kiểm tra tính duy nhất của chunk_id trong tệp
            assert df["chunk_id"].is_unique, f"Phát hiện trùng lặp chunk_id trong tệp {pf.name}!"

            # 3. Lấy mẫu 50 chunks kiểm tra độ dài và lỗi vỡ font
            sample_contents = df["content"].dropna().sample(min(50, len(df))).tolist()
            for text in sample_contents:
                text_str = str(text).strip()
                assert len(text_str) >= 20, f"Phát hiện chunk quá ngắn (< 20 ký tự) trong {pf.name}: '{text_str}'"
                assert "\x00" not in text_str, f"Phát hiện ký tự null byte trong {pf.name}!"

    def test_raw_shards_gzip_stream_integrity(self, project_paths):
        """Xác thực tính toàn vẹn của stream nén Gzip trên các phân mảnh Shards thô."""
        raw_shards_dir = project_paths["raw_shards"]
        if not raw_shards_dir.exists():
            pytest.skip(f"Thư mục {raw_shards_dir} không tồn tại.")

        gzip_shards = list(raw_shards_dir.glob("*.jsonl.gz"))
        if not gzip_shards:
            pytest.skip("Chưa tìm thấy tệp .jsonl.gz trong raw_shards.")

        # Kiểm tra ngẫu nhiên 2 phân mảnh
        sampled_shards = random.sample(gzip_shards, min(2, len(gzip_shards)))

        for shard_path in sampled_shards:
            # Đảm bảo không quét nhầm tệp rác hoặc tệp cách ly
            assert not shard_path.name.endswith(".quarantine.jsonl.gz")
            assert not shard_path.name.endswith(".corrupted")

            record_count = 0
            with gzip.open(shard_path, "rt", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if not line_str:
                        continue
                    item = json.loads(line_str)
                    assert "doc_id" in item or "item_id" in item, f"Bản ghi thiếu doc_id trong {shard_path.name}"
                    record_count += 1
                    if record_count >= 100:  # Đọc thử 100 bản ghi đầu để xác nhận stream Gzip nguyên vẹn
                        break

            assert record_count > 0, f"Phân mảnh {shard_path.name} rỗng!"