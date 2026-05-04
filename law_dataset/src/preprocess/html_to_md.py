import os
import sys
import glob
import html2text
from bs4 import BeautifulSoup
import logging
import re
from datetime import datetime

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN & LOGGING ĐỘNG
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Giả sử file nằm ở: law_dataset/src/preprocess/html_to_md.py
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../../"))

# Folder chứa dữ liệu
RAW_HTML_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'html')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Cấu hình Log theo ngày
TODAY_STR = datetime.now().strftime("%Y-%m-%d")
LOG_DIR = os.path.join(BASE_DIR, "data", "logs", TODAY_STR)
os.makedirs(LOG_DIR, exist_ok=True)

CURRENT_FILENAME = os.path.basename(__file__).split('.')[0]
LOG_FILE_PATH = os.path.join(LOG_DIR, f"log_{CURRENT_FILENAME}.log")

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, encoding="utf-8", mode="a"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class HTMLConverter:
    def __init__(self):
        # Cấu hình thư viện HTML2Text
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_links = True      # Bỏ qua link HTTP để sạch văn bản luật
        self.h2t.ignore_images = True     # Bỏ qua ảnh rác
        self.h2t.body_width = 0           # Không tự động cắt dòng
        self.h2t.protect_links = False

    def clean_html(self, html_content):
        """Dùng BeautifulSoup dọn dẹp các thẻ rác trước khi convert"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Xóa các thẻ rác của giao diện
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'noscript', 'button', 'iframe']):
            tag.decompose()

        # Tìm container chứa nội dung luật (VBPL chuẩn)
        main_content = soup.find('div', class_='fulltext') 
        if not main_content:
            main_content = soup.find('div', id='toanvancontent')
        if not main_content:
            main_content = soup.find('div', class_='content')
        if not main_content:
            main_content = soup.find('body')
            
        return str(main_content) if main_content else ""

    def normalize_legal_markdown(self, text):
        """Nắn chỉnh cấu trúc tuân thủ tuyệt đối Thể thức văn bản pháp luật VN"""
        
        # 1. Phần -> H1, Chương -> H2, Mục -> H3, Tiểu mục -> H4, Điều -> H5
        patterns = [
            (r'^(?:#|\*)*\s*(Phần\s+[IVXLCDM0-9]+)\s*[\.\:\*]*\s*(.*)$', r'# \1\n\2'),
            (r'^(?:#|\*)*\s*(Chương\s+[IVXLCDM0-9]+)\s*[\.\:\*]*\s*(.*)$', r'## \1\n\2'),
            (r'^(?:#|\*)*\s*(Mục\s+\d+)\s*[\.\:\*]*\s*(.*)$', r'### \1\n\2'),
            (r'^(?:#|\*)*\s*(Tiểu\s+mục\s+\d+)\s*[\.\:\*]*\s*(.*)$', r'#### \1\n\2'),
            (r'^(?:#|\*)*\s*(Điều\s+\d+[a-zA-Z]*)\s*[\.\:\*]*\s*(.*)$', r'##### \1\n\2'),
            (r'^(?:#|\*)*\s*(Phụ\s+lục\s*[IVXLCDM0-9]*)\s*[\.\:\*]*\s*(.*)$', r'# \1\n\2')
        ]
        
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text, flags=re.MULTILINE|re.IGNORECASE)

        # 2. Xử lý các số La Mã đứng lẻ (Thành H2)
        text = re.sub(r'^(?:#|\*)*\s*([IVXLCDM]+)\s*[\.\:\*]+\s*(.*)$', r'## \1. \2', text, flags=re.MULTILINE)

        # 3. Dọn dẹp dòng trống thừa
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text

    def convert_all(self):
        """Quét toàn bộ thư mục HTML và chuyển đổi hàng loạt"""
        html_files = glob.glob(os.path.join(RAW_HTML_DIR, '*.html'))
        total = len(html_files)
        
        if total == 0:
            logger.warning(f"Không tìm thấy file HTML nào trong: {RAW_HTML_DIR}")
            return

        logger.info(f"Bắt đầu chuyển đổi {total} file HTML sang Markdown...")
        
        for idx, file_path in enumerate(html_files):
            filename = os.path.basename(file_path)
            md_filename = filename.replace('.html', '.md')
            output_path = os.path.join(PROCESSED_DIR, md_filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                # Thực hiện quy trình 3 bước
                cleaned = self.clean_html(html_content)
                raw_md = self.h2t.handle(cleaned)
                final_md = self.normalize_legal_markdown(raw_md)
                
                with open(output_path, 'w', encoding='utf-8') as f_out:
                    f_out.write(final_md)
                
                logger.info(f"[{idx+1}/{total}] OK: {filename} -> {md_filename}")
                    
            except Exception as e:
                logger.error(f"Lỗi khi xử lý {filename}: {str(e)}")
                
        logger.info("Hoàn tất quá trình chuyển đổi HTML sang Markdown.")

if __name__ == "__main__":
    converter = HTMLConverter()
    converter.convert_all()