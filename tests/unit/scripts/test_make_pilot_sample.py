# -*- coding: utf-8 -*-
"""
Unit-test cho make_pilot_sample: doc_type_proxy (tái dùng hàm repo), allocate,
build_sample (tất định). Dữ liệu TỔNG HỢP, không DB/đĩa, không VLQA.
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for _p in (os.path.join(_REPO_ROOT, "scripts"), os.path.join(_REPO_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from make_pilot_sample import doc_type_proxy, allocate, build_sample  # noqa: E402


def test_doc_type_proxy_tai_dung_phan_loai_repo():
    assert doc_type_proxy("100/2019/NĐ-CP") == "Nghị định/Nghị quyết Chính phủ"
    assert doc_type_proxy("72/2026/QĐ-UBND") == "Văn bản địa phương"
    assert doc_type_proxy("15/2024/QH15") == "Luật/Nghị quyết Quốc hội"
    assert doc_type_proxy("123/2020/XYZ") == "Khác"           # VALID nhưng ký hiệu lạ → Khác
    assert doc_type_proxy("Không số") == "NO_NUMBER"
    assert doc_type_proxy("Nghị quyết về việc kỳ họp thứ 3 khóa 15") == "CONTAMINATED"
    assert doc_type_proxy("") == "CONTAMINATED"


def test_allocate_ti_le_min_per_va_clamp():
    alloc = allocate({"A": 1000, "B": 1000, "C": 10}, target=500, min_per=5)
    assert alloc["C"] == 5                      # tầng nhỏ được tối thiểu
    assert all(alloc[k] <= s for k, s in {"A": 1000, "B": 1000, "C": 10}.items())
    assert abs(sum(alloc.values()) - 500) <= 3  # xấp xỉ target (sai số làm tròn)
    assert alloc["A"] > 200 and alloc["B"] > 200 # tỉ lệ với cỡ tầng


def test_allocate_target_lon_hon_tong_lay_het():
    assert allocate({"A": 3, "B": 2}, target=100) == {"A": 3, "B": 2}


def _items():
    out = []
    for i in range(100):
        out.append({"doc_id": f"a{i}", "doc_number": f"{i}/2020/NĐ-CP",
                    "content_source": "detail_html",
                    "doc_type_proxy": "Nghị định/Nghị quyết Chính phủ",
                    "stratum": "detail_html|Nghị định/Nghị quyết Chính phủ"})
    for i in range(4):
        out.append({"doc_id": f"b{i}", "doc_number": "Không số",
                    "content_source": "fallback_html", "doc_type_proxy": "NO_NUMBER",
                    "stratum": "fallback_html|NO_NUMBER"})
    return out


def test_build_sample_tat_dinh_va_la_tap_con():
    items = _items()
    s1 = build_sample(items, target=50, min_per=5, seed=42)
    s2 = build_sample(items, target=50, min_per=5, seed=42)
    assert [x["doc_id"] for x in s1] == [x["doc_id"] for x in s2]   # tất định
    ids = {x["doc_id"] for x in items}
    assert all(x["doc_id"] in ids for x in s1)                      # tập con
    # tầng nhỏ (NO_NUMBER, 4 bản) phải xuất hiện đủ (min(min_per,4)=4)
    picked = [x for x in s1 if x["stratum"] == "fallback_html|NO_NUMBER"]
    assert len(picked) == 4
    # schema đầu ra
    assert set(s1[0]) >= {"doc_id", "doc_number", "doc_type_proxy", "content_source", "stratum"}
