import os
import glob
import html2text
from bs4 import BeautifulSoup
import logging
import re  # Thêm thư viện Regex

class HTMLConverter:
    def __init__(self):
        # Thiết lập đường dẫn động dựa trên vị trí file hiện tại
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.raw_html_dir = os.path.join(self.base_dir, '../../data/raw/html')
        self.processed_dir = os.path.join(self.base_dir, '../../data/processed')
        self.log_dir = os.path.join(self.base_dir, '../../data/logs')
        
        # Tự động tạo thư mục nếu chưa tồn tại
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Cấu hình Logger độc lập
        logging.basicConfig(
            filename=os.path.join(self.log_dir, 'preprocess.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger()
        
        # Cấu hình thư viện HTML2Text
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_links = True      # Bỏ qua link HTTP để sạch văn bản luật
        self.h2t.ignore_images = True     # Bỏ qua ảnh rác (nếu có)
        self.h2t.body_width = 0           # Rất quan trọng: Không tự động cắt dòng (wrap text)
        self.h2t.protect_links = False

    def clean_html(self, html_content):
        """Dùng BeautifulSoup dọn dẹp các thẻ rác trước khi convert"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Xóa ngay các thẻ rác của giao diện trước khi tìm nội dung
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'noscript', 'button', 'iframe']):
            tag.decompose()

        # Thử tìm các container chứa nội dung luật (VBPL hay dùng các class này)
        main_content = soup.find('div', class_='fulltext') 
        if not main_content:
            main_content = soup.find('div', id='toanvancontent')
        if not main_content:
            main_content = soup.find('div', class_='content')
            
        # Nếu vẫn không thấy thẻ div chuẩn, lấy toàn bộ body
        if not main_content:
            main_content = soup.find('body')
            
        if not main_content:
            return ""
        
        return str(main_content)

    def normalize_legal_markdown(self, text):
        """Nắn chỉnh cấu trúc tuân thủ tuyệt đối Thể thức văn bản pháp luật VN"""
        
        # 1. Bắt "Phần" -> Thành thẻ H1 (#)
        text = re.sub(r'^(?:#|\*)*\s*(Phần\s+[IVXLCDM0-9]+)\s*[\.\:\*]*\s*(.*)$', 
                      r'# \1\n\2', text, flags=re.MULTILINE|re.IGNORECASE)
        
        # 2. Bắt "Chương" -> Thành thẻ H2 (##)
        text = re.sub(r'^(?:#|\*)*\s*(Chương\s+[IVXLCDM0-9]+)\s*[\.\:\*]*\s*(.*)$', 
                      r'## \1\n\2', text, flags=re.MULTILINE|re.IGNORECASE)
        
        # 3. Bắt "Mục" -> Thành thẻ H3 (###)
        text = re.sub(r'^(?:#|\*)*\s*(Mục\s+\d+)\s*[\.\:\*]*\s*(.*)$', 
                      r'### \1\n\2', text, flags=re.MULTILINE|re.IGNORECASE)
        
        # 4. Bắt "Tiểu mục" -> Thành thẻ H4 (####)
        text = re.sub(r'^(?:#|\*)*\s*(Tiểu\s+mục\s+\d+)\s*[\.\:\*]*\s*(.*)$', 
                      r'#### \1\n\2', text, flags=re.MULTILINE|re.IGNORECASE)
        
        # 5. Bắt "Điều" -> Thành thẻ H5 (#####)
        text = re.sub(r'^(?:#|\*)*\s*(Điều\s+\d+[a-zA-Z]*)\s*[\.\:\*]*\s*(.*)$', 
                      r'##### \1\n\2', text, flags=re.MULTILINE|re.IGNORECASE)
                      
        # 6. Bắt "Phụ lục" -> Thành thẻ H1 (#) để nó là một khối độc lập ngang hàng với Phần
        text = re.sub(r'^(?:#|\*)*\s*(Phụ\s+lục\s*[IVXLCDM0-9]*)\s*[\.\:\*]*\s*(.*)$', 
                      r'# \1\n\2', text, flags=re.MULTILINE|re.IGNORECASE)

        # Xử lý các số La Mã đứng lẻ (dành cho Thông tư ngắn không có Chương) -> Thành H2 (##)
        text = re.sub(r'^(?:#|\*)*\s*([IVXLCDM]+)\s*[\.\:\*]+\s*(.*)$', 
                      r'## \1. \2', text, flags=re.MULTILINE)

        # Dọn dẹp dòng trống thừa
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text

    def convert_all(self):
        """Quét toàn bộ thư mục HTML và chuyển đổi hàng loạt sang Markdown chuẩn"""
        html_files = glob.glob(os.path.join(self.raw_html_dir, '*.html'))
        total = len(html_files)
        
        if total == 0:
            msg = "[THONG BAO] Khong tim thay file HTML nao trong data/raw/html/"
            print(msg)
            self.logger.warning(msg)
            return

        msg_start = f"[BAT DAU] Tim thay {total} file HTML. Dang tien hanh chuyen doi..."
        print(msg_start)
        self.logger.info(msg_start)
        
        for idx, file_path in enumerate(html_files):
            filename = os.path.basename(file_path)
            md_filename = filename.replace('.html', '.md')
            output_path = os.path.join(self.processed_dir, md_filename)
            
            try:
                # Đọc HTML
                with open(file_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                # Bước 1: Dọn dẹp HTML thô bằng BeautifulSoup
                clean_html = self.clean_html(html_content)
                
                # Bước 2: Convert sang Markdown thô bằng HTML2Text
                raw_md_text = self.h2t.handle(clean_html)
                
                # Bước 3: NẮN CHỈNH THÀNH MARKDOWN CHUẨN LUẬT (Khắc phục lỗi in đậm)
                final_md_text = self.normalize_legal_markdown(raw_md_text)
                
                # Bước 4: Ghi file MD (Overwrite nếu đã tồn tại)
                with open(output_path, 'w', encoding='utf-8') as f_out:
                    f_out.write(final_md_text)
                
                print(f"[{idx+1}/{total}] OK: {filename} -> {md_filename} (Đã nắn chỉnh Header)")
                self.logger.info(f"Thanh cong: {filename}")
                    
            except Exception as e:
                err_msg = f"[LOI] Xu ly {filename} that bai: {str(e)}"
                print(err_msg)
                self.logger.error(err_msg)
                
        print("[HOAN TAT] Da chuyen doi xong toan bo file HTML sang Markdown chuẩn!")

if __name__ == "__main__":
    converter = HTMLConverter()
    converter.convert_all()