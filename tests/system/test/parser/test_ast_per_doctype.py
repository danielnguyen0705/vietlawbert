"""
Tầng-1 — Kiểm phân cấp AST theo TỪNG LOẠI văn bản (cấu trúc nội tại biến thiên).
Chạy:  pytest tests/system/test/parser/test_ast_per_doctype.py -q
Parser thuần regex (không service) → test nhanh, tất định.
"""
import importlib.util, pathlib
_p = pathlib.Path(__file__).resolve().parents[4] / "preprocess" / "ast_parser.py"
_s = importlib.util.spec_from_file_location("ast_parser", _p)
AST = importlib.util.module_from_spec(_s); _s.loader.exec_module(AST)

def _parse(txt):
    return AST.HybridASTParser().parse_document(txt, {"doc_id": "D", "doc_number": "X"})
def _ids(c): return [x["chunk_id"] for x in c]
def _paths(c): return [x["hierarchy_path"] for x in c]

def test_hienphap_chuong_dieu_khong_phan():
    c = _parse("Chương I\nCHẾ ĐỘ CHÍNH TRỊ\nĐiều 1. Nước CHXHCN Việt Nam là một nước độc lập có chủ quyền.\nĐiều 2. Nhà nước pháp quyền xã hội chủ nghĩa của Nhân dân do Nhân dân.")
    assert len([x for x in c if x["article_num"] in ("1", "2")]) == 2
    assert all("Chương I" in p and "Phần" not in p for p in _paths(c))

def test_boluat_sau_phan_thu_tu_chu():
    # 'Phần thứ nhất' (Bộ luật Dân sự/Hình sự) + Chương + Mục + Điều + Khoản
    c = _parse("Phần thứ nhất\nQUY ĐỊNH CHUNG\nChương I\nNHỮNG QUY ĐỊNH CHUNG\nMục 1\nPHẠM VI\nĐiều 1. Phạm vi điều chỉnh\n1. Bộ luật này quy định địa vị pháp lý của cá nhân, pháp nhân.\n2. Quyền và nghĩa vụ về nhân thân và tài sản của cá nhân.")
    cl = [x for x in c if "_cl_" in x["chunk_id"]]
    assert len(cl) >= 2
    assert any(all(k in p for k in ["Phần", "Chương", "Mục", "Điều 1", "Khoản"]) for p in _paths(cl))

def test_thongtu_dieu_khoan_khong_chuong():
    c = _parse("Điều 1. Phạm vi điều chỉnh của thông tư này quy định rõ ràng cụ thể.\nĐiều 2. Đối tượng áp dụng\n1. Viên chức chuyên ngành công nghệ thông tin.\n2. Tổ chức cá nhân có liên quan đến việc áp dụng thông tư.")
    assert len([x for x in c if x["chunk_id"].endswith(("_art_2_cl_1", "_art_2_cl_2"))]) == 2

def test_quyetdinh_phang_chi_dieu():
    c = _parse("Điều 1. Ban hành kèm theo Quyết định này Quy chế quản lý nội bộ.\nĐiều 2. Quyết định này có hiệu lực kể từ ngày ký ban hành.\nĐiều 3. Chánh Văn phòng chịu trách nhiệm thi hành Quyết định.")
    assert len([x for x in c if x["chunk_id"] in ("D_art_1", "D_art_2", "D_art_3")]) == 3

def test_quyetdinh_kem_quyche_khong_va_cham_so_dieu():
    c = _parse("Điều 1. Ban hành kèm theo Quyết định này Quy chế tổ chức và hoạt động.\nĐiều 2. Quyết định này có hiệu lực kể từ ngày ký ban hành.\nQUY CHẾ\nTỔ CHỨC VÀ HOẠT ĐỘNG CỦA ĐƠN VỊ\nĐiều 1. Phạm vi điều chỉnh của Quy chế này áp dụng toàn đơn vị.\nĐiều 2. Nguyên tắc tổ chức và hoạt động của đơn vị trực thuộc.")
    I = _ids(c)
    assert "D_art_1" in I and "D_b1_art_1" in I and "D_b1_art_2" in I
    assert len(I) == len(set(I)), "chunk_id bị trùng giữa thân và phụ lục"

def test_preamble_can_cu_truoc_dieu_1():
    c = _parse("Căn cứ Luật Tổ chức Chính phủ ngày 19 tháng 6 năm 2015 đã được sửa đổi bổ sung;\nCăn cứ Nghị định số 01/2020/NĐ-CP quy định chi tiết thi hành một số điều của luật;\nĐiều 1. Phạm vi điều chỉnh của văn bản này được quy định rất rõ ràng cụ thể.")
    assert any("_preamble_" in x["chunk_id"] for x in c)
    assert any(x["chunk_id"] == "D_art_1" for x in c)
