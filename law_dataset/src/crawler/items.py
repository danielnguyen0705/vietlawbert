import scrapy

class VietLawItem(scrapy.Item):
    # --- Định danh cơ bản ---
    item_id = scrapy.Field()      # ID nội bộ của hệ thống VBPL (vd: 187505)
    doc_number = scrapy.Field()   # Số hiệu văn bản (vd: 80/2026/NĐ-CP)
    
    # --- Dữ liệu thô (Raw Data) ---
    full_text_html = scrapy.Field()  # Toàn bộ mã nguồn HTML của trang nội dung
    diagram_json = scrapy.Field()    # Dữ liệu JSON từ API Lược đồ (dành cho GraphRAG)
    
    # --- Siêu dữ liệu (Metadata) ---
    metadata_api = scrapy.Field()    # Lưu trọn bộ thông tin mà API tìm kiếm trả về (ngày ban hành, hiệu lực, cơ quan...)