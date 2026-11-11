# -*- coding: utf-8 -*-
"""
Unit-test cho các detector — fixture lấy TỪ CA THẬT đã xác minh tay trên mẫu Cochran 382.
Nguyên tắc dự án: làm đến đâu kiểm thử đến đó. Mỗi detector cần cả ca DƯƠNG (phải bắt)
lẫn ca ÂM (không được báo nhầm văn bản sạch).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detectors import (  # noqa: E402
    Chunk, detect_ocr_corruption, detect_execution_tail, detect_empty_node,
    detect_flattened_table, detect_doctype_mismatch, detect_hierarchy_anomaly,
    detect_identity_collision_risk,
)

# ---- Ca sạch để kiểm ÂM (tránh false positive) ---------------------------
CLEAN_BODY = ("Điều 15. Quản lý, sử dụng chữ ký số chuyên dùng. "
              "1. Chữ ký số chuyên dùng công vụ sử dụng trong hoạt động công vụ "
              "của các cơ quan, đơn vị được quản lý theo quy định của pháp luật.")


# ---- #1 OCR vỡ -----------------------------------------------------------
def test_ocr_bat_duoc_rac_that():
    body = "- O6i vOi t6 chric: T€n t6 chric, <lia chi tru sd chinh, sii Ci6y phdp thAnh lflp"
    flag, score = detect_ocr_corruption(body)
    assert flag and score >= 0.12


def test_ocr_khong_bao_nham_van_ban_sach():
    flag, _ = detect_ocr_corruption(CLEAN_BODY)
    assert not flag


def test_ocr_khong_bao_nham_so_hieu_va_so_lieu():
    # số liệu/định mức + tham chiếu luật không được coi là OCR vỡ
    flag, _ = detect_ocr_corruption("Bảng 32: định mức 100,00% theo Thông tư 15/2020/TT-BTC, giấy A4, ổ 2TB")
    assert not flag


# ---- #2 đuôi chữ ký / thi hành -------------------------------------------
def test_execution_tail_bat_khoi_thi_hanh():
    body = ("Điều 7. - Chánh Văn phòng ... chịu trách nhiệm thi hành quyết định này./.\n"
            "T/M ỦY BAN NHÂN DÂN THÀNH PHỐ\nK/T CHỦ TỊCH\nPHÓ CHỦ TỊCH\nNguyễn Văn Chí")
    flag, _ = detect_execution_tail(body)
    assert flag


def test_execution_tail_khong_bat_noi_dung_quy_pham():
    flag, _ = detect_execution_tail(CLEAN_BODY)
    assert not flag


# ---- #3 node cụt ---------------------------------------------------------
def test_empty_node_bat_node_chi_con_tieu_de():
    body = "CHẾ ĐỘ BÁO CÁO\nĐiều 5. Nội dung báo cáo"
    flag, n = detect_empty_node(body)
    assert flag and n < 50


def test_empty_node_khong_bat_dieu_co_than():
    flag, _ = detect_empty_node(CLEAN_BODY)
    assert not flag


# ---- #4 bảng dẹt ---------------------------------------------------------
def test_flattened_table_bat_bang_bi_det():
    body = ("1. Cơ cấu định mức như sau:\nSTT\nHạng mục\nCơ cấu (%)\nNội nghiệp\n"
            "Ngoại nghiệp\nTổng số\n100,00\n100,00\n1\nĐiều tra\n6,11")
    flag, _ = detect_flattened_table(body)
    assert flag


def test_flattened_table_khong_bat_van_xuoi():
    flag, _ = detect_flattened_table(CLEAN_BODY)
    assert not flag


# ---- #5 doc_type sai -----------------------------------------------------
def test_doctype_mismatch_thong_tu_bi_gan_khac():
    meta = ("Văn bản: 60-TC/CÐKT | Tiêu đề: Thông tư số 60-TC/CÐKT hướng dẫn công tác "
            "kế toán | Cơ quan: Bộ Tài chính")
    flag, _ = detect_doctype_mismatch(meta, "Khác")
    assert flag


def test_doctype_vbhn_khong_phat_vi_khong_co_bucket():
    # VBHN không có bucket riêng trong taxonomy → VBHN→Khác KHÔNG bị coi là lỗi
    meta = "Tiêu đề: Văn bản hợp nhất số 30/VBHN-NHNN Quy định về hồ sơ | Cơ quan: NHNN"
    flag, _ = detect_doctype_mismatch(meta, "Khác", "30/VBHN-NHNN")
    assert not flag


def test_doctype_dia_phuong_la_dung_khong_bao():
    # QĐ-UBND xếp 'Văn bản địa phương' là ĐÚNG taxonomy (phân theo cấp ban hành)
    meta = "Tiêu đề: Quyết định số 17/2012/QĐ-UBND ... | Cơ quan: UBND tỉnh"
    flag, _ = detect_doctype_mismatch(meta, "Văn bản địa phương", "17/2012/QĐ-UBND")
    assert not flag


def test_doctype_khop_dung_khong_bao():
    meta = "Tiêu đề: Thông tư số 03/2015/TT-BYT Quy định chế độ báo cáo | Cơ quan: Bộ Y tế"
    flag, _ = detect_doctype_mismatch(meta, "Thông tư", "03/2015/TT-BYT")
    assert not flag


# ---- #6 hierarchy tùy hứng ----------------------------------------------
def test_hierarchy_cap_phi_cau_truc_tren_chunk_thuong():
    # 'QUY ĐỊNH CHUNG' đứng làm một cấp trên chunk điều (không phải preamble) = bất thường
    flag, _ = detect_hierarchy_anomaly("QUY ĐỊNH CHUNG", "Điều 3. ...", "x_art_3")
    assert flag


def test_hierarchy_preamble_can_cu_khong_phat():
    # preamble mang nhãn 'Căn cứ pháp lý' là nhất quán, KHÔNG phạt
    flag, _ = detect_hierarchy_anomaly("Căn cứ pháp lý", "- Đối với tổ chức: ...", "x_preamble_3")
    assert not flag


def test_hierarchy_dieu_lech_body():
    flag, _ = detect_hierarchy_anomaly("Chương II > Điều 4", "Điều 5. Nội dung báo cáo")
    assert flag


def test_hierarchy_hop_le_khong_bao():
    flag, _ = detect_hierarchy_anomaly("Chương II > Điều 5 > Khoản 1", "Điều 5. 1. Nội dung ...")
    assert not flag


# ---- #7 rủi ro trùng khóa ------------------------------------------------
def test_collision_risk_vbhn():
    flag, _ = detect_identity_collision_risk("30/VBHN-NHNN")
    assert flag


def test_collision_risk_so_hieu_thuong_khong_bao():
    flag, _ = detect_identity_collision_risk("03/2015/TT-BYT")
    assert not flag


# ---- gộp qua Chunk.from_flat --------------------------------------------
def test_chunk_from_flat_tach_dung_3_truong():
    content = ("[META] Văn bản: 03/2015/TT-BYT | Tiêu đề: Thông tư số 03/2015/TT-BYT\n"
               "[HIERARCHY] Chương II\n[CONTENT] CHẾ ĐỘ BÁO CÁO\nĐiều 5. Nội dung báo cáo")
    c = Chunk.from_flat("x_art_4_root", "03/2015/TT-BYT", "Thông tư", content)
    assert "03/2015/TT-BYT" in c.meta
    assert c.hierarchy == "Chương II"
    assert c.body.startswith("CHẾ ĐỘ BÁO CÁO")
