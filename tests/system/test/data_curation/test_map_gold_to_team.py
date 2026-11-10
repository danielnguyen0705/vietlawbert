"""
tests/system/test/data_curation/test_map_gold_to_team.py — Kiểm thử đơn vị cho việc ánh xạ gold sang chunk nhóm.
"""
from __future__ import annotations

import os
import sys
import pytest

# Bỏ _common để tránh xung đột gói datasets
_removed = [p for p in sys.path if os.path.basename(os.path.normpath(p)) == "_common"]
for p in _removed:
    sys.path.remove(p)

from benchmark.map_gold_to_team import match_article, map_vlb_item

for p in _removed:
    sys.path.append(p)

pytestmark = pytest.mark.tier1

def test_match_article_single():
    article = "Điều 1. Phạm vi điều chỉnh và đối tượng áp dụng của quy định này."
    candidates = [
        {
            "chunk_id": "c1",
            "doc_id": "d1",
            "doc_number": "01/2026/QD-UBND",
            "content": "Điều 1. Phạm vi điều chỉnh và đối tượng áp dụng của quy định này.",
            "level": "ARTICLE"
        }
    ]
    res = match_article(article, candidates)
    assert res["decision"] == "single"
    assert res["gold_chunk_ids"] == ["c1"]
    assert res["best_score"] == pytest.approx(1.0)

def test_match_article_split():
    part1 = "Điều 2. Nguyên tắc tổ chức và hoạt động của cơ quan nhà nước gồm có minh bạch,"
    part2 = "công khai dân chủ và bảo đảm quyền con người quyền công dân theo quy định pháp luật."
    article = part1 + " " + part2

    candidates = [
        {
            "chunk_id": "c1",
            "doc_id": "d1",
            "doc_number": "01/2026/QD-UBND",
            "content": part1,
            "level": "ARTICLE"
        },
        {
            "chunk_id": "c2",
            "doc_id": "d1",
            "doc_number": "01/2026/QD-UBND",
            "content": part2,
            "level": "ARTICLE"
        }
    ]
    res = match_article(article, candidates)
    assert res["decision"] == "split"
    assert sorted(res["gold_chunk_ids"]) == ["c1", "c2"]

def test_match_article_duplicate_copies():
    article = "Điều 3. Trách nhiệm của các cơ quan tổ chức cá nhân có liên quan trong quản lý."
    candidates = [
        {
            "chunk_id": "c1_d1",
            "doc_id": "d1",
            "doc_number": "01/2026/QD-UBND",
            "content": article,
            "level": "ARTICLE"
        },
        {
            "chunk_id": "c1_d2",
            "doc_id": "d2",  # Doc_id khác nhưng cùng số hiệu
            "doc_number": "01/2026/QD-UBND",
            "content": article,
            "level": "ARTICLE"
        }
    ]
    res = match_article(article, candidates)
    assert res["decision"] == "duplicate copies"
    assert "c1_d1" in res["gold_chunk_ids"]
    assert "c1_d2" in res["gold_chunk_ids"]
    assert len(res["gold_chunk_ids"]) == 2

def test_match_article_weak():
    # Khoảng 60% trùng lặp n-gram
    article = "Điều 4. Tiêu chuẩn và điều kiện bổ nhiệm công chức lãnh đạo quản lý cấp phòng và tương đương."
    candidate_content = "Điều 4. Tiêu chuẩn và điều kiện bổ nhiệm cán bộ công nhân viên chức sự nghiệp y tế."
    candidates = [
        {
            "chunk_id": "c1",
            "doc_id": "d1",
            "doc_number": "01/2026/QD-UBND",
            "content": candidate_content,
            "level": "ARTICLE"
        }
    ]
    res = match_article(article, candidates)
    assert res["decision"] == "weak"
    assert res["gold_chunk_ids"] == []
    assert 0.50 <= res["best_score"] < 0.80

def test_match_article_no_match_and_missing_doc():
    article = "Điều 5. Thẩm quyền quyết định việc phê duyệt đề án bảo vệ môi trường."
    unrelated_candidates = [
        {
            "chunk_id": "c1",
            "doc_id": "d1",
            "doc_number": "01/2026/QD-UBND",
            "content": "Hoàn toàn khác biệt xyz1234567890 không có từ ngữ nào tương tự.",
            "level": "ARTICLE"
        }
    ]
    res_no_match = match_article(article, unrelated_candidates)
    assert res_no_match["decision"] == "no_match"
    assert res_no_match["gold_chunk_ids"] == []

    # Không có candidate nào
    res_missing = match_article(article, [])
    assert res_missing["decision"] == "missing_doc"
    assert res_missing["gold_chunk_ids"] == []

def test_map_vlb_row():
    # Item giả lập VietLawBench
    item = {
        "benchmark_id": "vlb_9999",
        "query": "Quy định về thời hạn tạm giữ?",
        "query_type": "single_hop",
        "ground_truth_chunk": "D1_art_6_cl_5",
        "ground_truth_doc_number": "58/2026/TT-BCA",
        "ground_truth_article": "Điều 6",
        "evidence_text": "Thời hạn tạm giữ không được quá ba ngày kể từ khi bắt giữ người phạm tội quả tang."
    }

    # Candidates giả lập: có chunk D1_art_6 chứa evidence_text và D1_art_7 khác
    candidates = [
        {
            "chunk_id": "D1_art_6",
            "doc_id": "D1",
            "doc_number": "58/2026/TT-BCA",
            "hierarchy_path": "Thông tư 58 > Điều 6.",
            "content": "Điều 6. Tạm giữ người phạm pháp\n1. Thời hạn tạm giữ không được quá ba ngày kể từ khi bắt giữ người phạm tội quả tang.\n2. Các trường hợp khác theo luật.",
            "level": "ARTICLE"
        },
        {
            "chunk_id": "D1_art_7",
            "doc_id": "D1",
            "doc_number": "58/2026/TT-BCA",
            "hierarchy_path": "Thông tư 58 > Điều 7.",
            "content": "Điều 7. Tạm giam người có hành vi vi phạm pháp luật hình sự.",
            "level": "ARTICLE"
        }
    ]

    res = map_vlb_item(item, candidates)
    assert res["status"] == "mapped"
    assert res["relevant_chunk_ids"] == ["D1_art_6"]
    assert res["evidence_containment"] >= 0.80

    # Trường hợp evidence_text không khớp
    wrong_evidence_item = dict(item)
    wrong_evidence_item["evidence_text"] = "Nội dung hoàn toàn sai lệch không có trong điều 6 abcxyz."
    res_mismatch = map_vlb_item(wrong_evidence_item, candidates)
    assert res_mismatch["status"] == "text_mismatch"
    assert res_mismatch["relevant_chunk_ids"] == []