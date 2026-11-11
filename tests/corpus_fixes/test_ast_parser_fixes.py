# -*- coding: utf-8 -*-
"""Kiểm ast_parser: reset số Điều tại ranh giới chương; không dán nhãn 'Căn cứ' cho dòng mồ côi
giữa văn bản; re_attached không nuốt 'QUY ĐỊNH CHUNG'."""
from preprocess.ast_parser import HybridASTParser

META = {"doc_id": "D1", "doc_number": "03/2015/TT-BYT", "title": "Thông tư 03/2015/TT-BYT",
        "co_quan": "Bộ Y tế", "effective_date": "2015-04-29", "status": "Hết hiệu lực toàn bộ"}


def _parse(text):
    return HybridASTParser().parse_document(text, META)


def test_ranh_gioi_chuong_dong_dieu_truoc():
    # Lỗi gốc: 'CHẾ ĐỘ BÁO CÁO' + 'Điều 5' bị gán vào số Điều 4. Sau sửa: KHÔNG còn.
    text = ("Điều 4. Trách nhiệm báo cáo\n"
            "Các đơn vị có trách nhiệm báo cáo đầy đủ theo quy định của Thông tư này.\n"
            "Chương II\n"
            "CHẾ ĐỘ BÁO CÁO\n"
            "Điều 5. Nội dung báo cáo\n"
            "1. Báo cáo định kỳ gồm các nội dung về tình hình phòng chống dịch bệnh.\n")
    chunks = _parse(text)
    by_art = {c["article_num"]: c for c in chunks if c["article_num"]}
    assert "5" in by_art                                   # Điều 5 được nhận đúng
    art4 = [c for c in chunks if c["article_num"] == "4"]
    # không chunk nào của Điều 4 chứa 'Điều 5' hay tiêu đề chương
    assert all("Điều 5" not in c["content"] and "CHẾ ĐỘ BÁO CÁO" not in c["content"] for c in art4)


def test_dong_mo_coi_sau_chuong_khong_bi_dan_nhan_can_cu():
    # Dòng dài sau tiêu đề chương (không còn Điều hiện hành) KHÔNG được gắn 'Căn cứ pháp lý'.
    text = ("Điều 1. Phạm vi điều chỉnh\nThông tư này quy định chế độ báo cáo công tác.\n"
            "Chương II\n"
            "Đây là một đoạn dẫn nhập dài của chương hai nói về phạm vi áp dụng chế độ báo cáo định kỳ.\n"
            "Điều 2. Đối tượng áp dụng\nÁp dụng cho mọi đơn vị trực thuộc theo phân cấp quản lý.\n")
    chunks = _parse(text)
    orphan = [c for c in chunks if "dẫn nhập dài của chương hai" in c["content"]]
    assert orphan, "đoạn mồ côi phải được giữ lại"
    assert "Căn cứ pháp lý" not in orphan[0]["content"]    # không dán nhãn căn cứ tùy hứng
    assert "Chương II" in orphan[0]["hierarchy_path"]       # gắn đúng ngữ cảnh chương


def test_preamble_truoc_dieu_dau_van_la_can_cu():
    text = ("Căn cứ Luật Ban hành văn bản quy phạm pháp luật năm 2025 và các văn bản liên quan;\n"
            "Điều 1. Phạm vi\nThông tư này quy định về chế độ báo cáo định kỳ công tác.\n")
    chunks = _parse(text)
    pre = [c for c in chunks if "_preamble_" in c["chunk_id"]]
    assert pre and "Căn cứ pháp lý" in pre[0]["content"]    # tiền văn thật vẫn là căn cứ


def test_re_attached_khong_nuot_quy_dinh_chung():
    text = ("Điều 1. Ban hành kèm theo Quyết định này Quy định về quản lý.\n"
            "QUY ĐỊNH CHUNG\n"
            "Điều 2. Giải thích từ ngữ\nTrong Quy định này các từ ngữ được hiểu như sau đây.\n")
    chunks = _parse(text)
    # 'QUY ĐỊNH CHUNG' KHÔNG được trở thành một cấp chương trong hierarchy
    assert all("QUY ĐỊNH CHUNG" not in c["hierarchy_path"] for c in chunks)
