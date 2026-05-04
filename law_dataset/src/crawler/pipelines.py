import os
import json
import logging
from utils.helpers import get_clean_filename

class VietLawDataPipeline:
    def __init__(self):
        # Cấu hình các đường dẫn tương đối (Nhảy ra khỏi src/crawler để vào data/raw)
        self.raw_dir = os.path.join(os.path.dirname(__file__), '../../data/raw')
        self.html_dir = os.path.join(self.raw_dir, 'html')
        self.diagram_dir = os.path.join(self.raw_dir, 'diagram')
        self.metadata_file = os.path.join(self.raw_dir, 'metadata.jsonl')

    def open_spider(self, spider):
        """Hàm này tự động chạy KHI CON NHỆN BẮT ĐẦU khởi động"""
        spider.logger.info("Đang thiết lập thư mục lưu trữ...")
        
        # Tự động tạo thư mục nếu chưa tồn tại
        os.makedirs(self.html_dir, exist_ok=True)
        os.makedirs(self.diagram_dir, exist_ok=True)
        
        # Mở file metadata.jsonl ở chế độ 'a' (append - ghi nối tiếp)
        self.file = open(self.metadata_file, 'a', encoding='utf-8')

    def close_spider(self, spider):
        """Hàm này tự động chạy KHI CON NHỆN KẾT THÚC"""
        spider.logger.info("Đóng file lưu trữ. Kết thúc chiến dịch!")
        if self.file and not self.file.closed:
            self.file.close()

    def process_item(self, item, spider):
        """Hàm này xử lý TỪNG VĂN BẢN một khi Spider gửi về"""
        try:
            # 1. Tạo tên file chuẩn chỉ dùng ID (ví dụ: 173920)
            item_id = item.get('item_id', 'unknown_id')
            filename = get_clean_filename(None, item_id)

            # 2. LƯU FILE HTML TOÀN VĂN
            html_content = item.get('full_text_html')
            if html_content:
                html_path = os.path.join(self.html_dir, f"{filename}.html")
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)

            # 3. LƯU FILE JSON LƯỢC ĐỒ (Dành cho Neo4j sau này)
            diagram_data = item.get('diagram_json')
            if diagram_data:
                diagram_path = os.path.join(self.diagram_dir, f"{filename}.json")
                with open(diagram_path, 'w', encoding='utf-8') as f:
                    json.dump(diagram_data, f, ensure_ascii=False, indent=4)

            # --- FIX LỖI CRASH KHI METADATA BỊ RỖNG ---
            # Nếu item.get('metadata_api') là None, ép nó thành dictionary rỗng {}
            meta_api = item.get('metadata_api') or {}

            # 4. LƯU METADATA VÀO JSONL (Dành cho Milvus/MongoDB sau này)
            metadata_record = {
                "id": filename,  
                "doc_number": item.get('doc_number', 'Unknown'),
                "title": meta_api.get('title', 'Unknown'),
                "issue_date": meta_api.get('issueDate', ''),
                "file_html": f"{filename}.html",
                "file_diagram": f"{filename}.json",
                "metadata_api": meta_api  
            }
            
            # Ghi 1 dòng JSON vào file .jsonl
            self.file.write(json.dumps(metadata_record, ensure_ascii=False) + '\n')
            self.file.flush() 

        except Exception as e:
            spider.logger.error(f"[PIPELINE LỖI] Không thể lưu item {item.get('item_id')}: {str(e)}")

        return item