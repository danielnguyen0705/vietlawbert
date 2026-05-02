def get_clean_filename(doc_number, item_id):
    """
    Thống nhất dùng ID hệ thống làm tên file để dễ quản lý GraphRAG.
    Ví dụ: item_id '173920' -> filename '173920'
    """
    if not item_id:
        # Dự phòng trường hợp API lỗi không trả về ID
        return "Unknown_ID"
    
    return str(item_id).strip()

# (Tuỳ chọn: Nếu sau này bạn cần hàm làm sạch text riêng biệt, hãy thêm vào đây)