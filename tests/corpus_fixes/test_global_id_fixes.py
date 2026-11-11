# -*- coding: utf-8 -*-
"""Kiểm doc_type_and_rank (quét toàn ký hiệu + cứu tiêu đề) và canonical_doc_id disambiguator."""
from configs.global_id import (doc_type_and_rank, canonical_doc_id, detect_collisions,
                               compute_record_doc_id, resolve_target_id, is_collision_prone)


def _rec(dn, issue="", org=""):
    return {"doc_number": dn, "issue_date": issue, "co_quan": org}


# --- doc_type: các ca ĐÃ xác minh tay trên mẫu Cochran bị gán sai 'Khác' ---
def test_thong_tu_bo_format_cu_60_TC():
    assert doc_type_and_rank("60-TC/CÐKT", "Thông tư số 60-TC/CÐKT hướng dẫn kế toán")[0] == "Thông tư"

def test_thong_tu_bo_format_cu_05_BXD_tu_so_hieu():
    # ngay cả KHÔNG có tiêu đề, định dạng bộ cũ <num>-<BO>/<DIV> suy ra Thông tư
    assert doc_type_and_rank("05-BXD/KTQH")[0] == "Thông tư"

def test_quyet_dinh_ub_vx_la_dia_phuong():
    assert doc_type_and_rank("3491/QĐ-UB-VX")[0] == "Văn bản địa phương"

def test_hdbt_la_chinh_phu():
    assert doc_type_and_rank("84/HĐBT")[0] == "Nghị định/Nghị quyết Chính phủ"

def test_vbhn_la_van_ban_hop_nhat():
    dt, rank = doc_type_and_rank("30/VBHN-NHNN", "Văn bản hợp nhất số 30/VBHN-NHNN")
    assert dt == "Văn bản hợp nhất"

def test_luat_format_cu_58_L_CTN_qua_tieu_de():
    # 58-L/CTN: ký hiệu dễ nhầm Chủ tịch nước, nhưng tiêu đề 'Luật' là hình thức đúng
    assert doc_type_and_rank("58-L/CTN", "Luật số 58-L/CTN về ...")[0] == "Luật/Nghị quyết Quốc hội"

# --- không được báo nhầm: các ca ĐÚNG sẵn ---
def test_qd_ubnd_van_la_dia_phuong():
    assert doc_type_and_rank("17/2012/QĐ-UBND", "Quyết định số 17/2012/QĐ-UBND")[0] == "Văn bản địa phương"

def test_thong_tu_hien_dai_khong_lech():
    assert doc_type_and_rank("03/2015/TT-BYT", "Thông tư số 03/2015/TT-BYT")[0] == "Thông tư"

def test_nghi_dinh_khong_bi_nham_luat_trong_tieu_de():
    # tiêu đề có chữ 'Luật' ở GIỮA không được biến Nghị định thành Luật
    dt = doc_type_and_rank("47/2020/NĐ-CP", "Nghị định quy định chi tiết một số điều của Luật Doanh nghiệp")[0]
    assert dt == "Nghị định/Nghị quyết Chính phủ"

def test_qh_van_la_luat():
    assert doc_type_and_rank("70/2018/QH14")[0] == "Luật/Nghị quyết Quốc hội"


# --- canonical_doc_id: chống trùng số hiệu lặp (VBHN theo năm) ---
def test_canonical_tuong_thich_nguoc_khong_disambiguator():
    # không cấp ngày/cơ quan → giữ hành vi CŨ (hash theo slug) để không phá khớp cạnh
    a = canonical_doc_id("30/VBHN-NHNN")
    b = canonical_doc_id("30/VBHN-NHNN")
    assert a == b and a.startswith("30-VBHN-NHNN-")

def test_canonical_disambiguator_tach_ban_trung_so_hieu():
    a = canonical_doc_id("30/VBHN-NHNN", sign_date="2024-08-21", issuer="NHNN")
    b = canonical_doc_id("30/VBHN-NHNN", sign_date="2016-07-09", issuer="NHNN")
    assert a != b                                  # cùng số hiệu, khác ngày → KHÁC id (không gộp)
    assert a.startswith("30-VBHN-NHNN-") and b.startswith("30-VBHN-NHNN-")  # slug đọc được giữ nguyên

def test_is_collision_prone_chi_vbhn():
    assert is_collision_prone("30/VBHN-NHNN")
    assert not is_collision_prone("03/2015/TT-BYT")


def test_compute_record_vbhn_hai_ban_khac_ngay_khac_id():
    a = compute_record_doc_id(_rec("30/VBHN-NHNN", "2024-08-21", "NHNN"))
    b = compute_record_doc_id(_rec("30/VBHN-NHNN", "2016-07-09", "NHNN"))
    assert a and b and a != b                        # 2 bản VBHN KHÔNG gộp


def test_compute_record_non_vbhn_tuong_thich_nguoc():
    # số hiệu thường: version KHÔNG ảnh hưởng → id y hệt canonical trần ⇒ không phá khớp cạnh/ES cũ
    rid = compute_record_doc_id(_rec("03/2015/TT-BYT", "2015-04-29", "Bộ Y tế"))
    assert rid == canonical_doc_id("03/2015/TT-BYT")


def test_resolve_vbhn_tro_ban_moi_nhat():
    newer = compute_record_doc_id(_rec("30/VBHN-NHNN", "2024-08-21", "NHNN"))
    older = compute_record_doc_id(_rec("30/VBHN-NHNN", "2016-07-09", "NHNN"))
    slug = newer.rsplit("-", 1)[0]
    latest = {slug: newer}
    got = resolve_target_id("30/VBHN-NHNN", {}, latest)   # cạnh chỉ có số hiệu trần
    assert got == newer and got != older                  # trỏ bản mới nhất, không phải bản cũ/plain


def test_resolve_non_vbhn_khong_doi():
    assert resolve_target_id("03/2015/TT-BYT", {}, {"x": "y"}) == canonical_doc_id("03/2015/TT-BYT")


def test_detect_collisions_bat_so_hieu_lap():
    recs = [
        {"doc_number": "30/VBHN-NHNN", "sign_date": "2024-08-21"},
        {"doc_number": "30/VBHN-NHNN", "sign_date": "2016-07-09"},
        {"doc_number": "03/2015/TT-BYT", "sign_date": "2015-04-29"},
    ]
    col = detect_collisions(recs)
    slugs = list(col.keys())
    assert any("30-VBHN-NHNN" in s for s in slugs)
    assert not any("03-2015-TT-BYT" in s for s in slugs)
