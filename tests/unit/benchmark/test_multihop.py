# -*- coding: utf-8 -*-
"""Self-check các cổng thuần của build_multihop_subset (không gọi LLM/Neo4j)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src", "benchmark"))
import build_multihop_subset as M  # noqa: E402


def test_doc_of_chunk():
    assert M.doc_of_chunk("abc-uuid_art_2") == "abc-uuid"
    assert M.doc_of_chunk("abc-uuid_art_2_cl_3") == "abc-uuid"
    assert M.doc_of_chunk("") == ""


def test_is_meta_va_procedural():
    assert M.is_meta("[META] Văn bản: 10/2023") is True
    assert M.is_meta("1. Quy định về xử phạt") is False
    assert M.is_procedural("Điều 20. Hiệu lực thi hành Thông tư này có hiệu lực từ...") is True
    assert M.is_procedural("Điều 5. Tổ chức thực hiện") is True
    assert M.is_procedural("1. Người điều khiển xe mô tô vượt đèn đỏ bị phạt tiền") is False


def test_strip_headers():
    raw = ("[META] Văn bản: 68/2026/QĐ-CTUBND | Tiêu đề: ... | Tình trạng: Còn hiệu lực\n"
           "[HIERARCHY] Điều 1 (Sửa đổi, bổ sung Điều 1)\n"
           "[CONTENT] Người điều khiển xe mô tô vượt đèn đỏ bị phạt tiền.")
    body = M.strip_headers(raw)
    assert body == "Người điều khiển xe mô tô vượt đèn đỏ bị phạt tiền."
    assert "[META]" not in body and "68/2026" not in body     # số hiệu trong META bị cắt
    assert M.strip_headers("Không có header, chỉ thân.") == "Không có header, chỉ thân."


def test_jaccard_va_pick_pair():
    assert M.jaccard("phạt tiền vượt đèn đỏ", "phạt tiền vượt đèn đỏ") == 1.0
    assert M.jaccard("thuế thu nhập doanh nghiệp", "an toàn giao thông đường bộ") == 0.0
    a = [{"chunk_id": "A1", "content": "mức phạt tiền vi phạm giao thông đường bộ"},
         {"chunk_id": "A2", "content": "quy định về thuế giá trị gia tăng"}]
    b = [{"chunk_id": "B1", "content": "sửa đổi mức phạt tiền vi phạm giao thông đường bộ"}]
    ca, cb, s = M.pick_best_chunk_pair(a, b)
    assert ca["chunk_id"] == "A1" and cb["chunk_id"] == "B1" and s > 0.3
    # cặp khác chủ đề hẳn → điểm dưới ngưỡng MIN_JACCARD (bị loại ở build)
    d = [{"chunk_id": "X", "content": "thuế thu nhập doanh nghiệp kê khai nộp ngân sách"}]
    e = [{"chunk_id": "Y", "content": "an toàn giao thông mũ bảo hiểm xe máy"}]
    assert M.pick_best_chunk_pair(d, e)[2] < M.MIN_JACCARD


def test_leaks_doc_number():
    assert M.leaks_doc_number("Theo Nghị định 10/2023/NĐ-CP thì sao?", "10/2023/NĐ-CP") is True
    assert M.leaks_doc_number("Doanh nghiệp phải nộp giấy tờ gì khi hoàn thành công trình?",
                              "10/2023/NĐ-CP", "43/2014/NĐ-CP") is False
    assert M.leaks_doc_number("Quy định tại 10/2023 ra sao", "10/2023/NĐ-CP") is True   # lõi số/năm


def test_load_leakage_pairs(tmp_path):
    import pandas as pd
    p = tmp_path / "trip.parquet"
    pd.DataFrame({
        "anchor_id": ["docA_art_1", "docA_art_2", "docC_art_1"],
        "positive_id": ["docB_art_3", "docA_art_5", "docD_art_2"],  # dòng 2 cùng doc -> bỏ
    }).to_parquet(p)
    pairs, info = M.load_leakage_pairs(p)      # không Neo4j -> khóa theo doc_id
    assert frozenset(("docA", "docB")) in pairs
    assert frozenset(("docC", "docD")) in pairs
    assert frozenset(("docA",)) not in pairs        # cùng doc không tính
    assert len(pairs) == 2
