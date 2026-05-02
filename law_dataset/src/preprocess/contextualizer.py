import os
import sys
import json
import time
import logging
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter
from google import genai
from google.genai import types

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN & LOGGING
# ==========================================
# Lấy đường dẫn thư mục 'law_dataset'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MD_FOLDER = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "json")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "final_contextual_chunks.jsonl")

# Cấu hình thư mục Log
LOG_DIR = os.path.join(BASE_DIR, "data", "log")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Lấy tên file hiện tại (bỏ đuôi .py) để đặt tên log
current_filename = os.path.basename(__file__).split('.')[0]
log_filepath = os.path.join(LOG_DIR, f"log_{current_filename}.log")

# Thiết lập bộ ghi log (Vừa ghi ra file, vừa in ra Terminal)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filepath, encoding="utf-8", mode="a"), # Ghi nối đuôi vào file
        logging.StreamHandler(sys.stdout) # In ra màn hình Terminal
    ]
)
logger = logging.getLogger(__name__)

# ==========================================
# 2. KHỞI TẠO API TỪ FILE .ENV 
# ==========================================
env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(env_path)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    parent_env = os.path.join(os.path.dirname(BASE_DIR), ".env")
    load_dotenv(parent_env)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error(f"Không tìm thấy GEMINI_API_KEY trong file .env ở cả {env_path} và {parent_env}")
        raise ValueError("Thiếu API Key.")

# Khởi tạo Client bằng SDK mới
client = genai.Client(api_key=api_key)
# Nhớ kiểm tra lại tên model cho đúng nhé 
MODEL_ID = 'gemini-3.1-pro' 

# ==========================================
# 3. CẤU HÌNH BỘ CẮT MARKDOWN
# ==========================================
headers_to_split_on = [
    ("#", "Phần / Phụ lục"), 
    ("##", "Chương"),         
    ("###", "Mục"),        
    ("####", "Tiểu mục"),   
    ("#####", "Điều")       
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

def generate_context(chunk_content, document_title):
    """Gọi Gemini bằng SDK mới để sinh ngữ cảnh"""
    prompt = f"""
    Bạn là một chuyên gia pháp lý. Hãy viết 1 câu ngữ cảnh giải thích ngắn gọn cho đoạn văn bản luật sau.
    Tên văn bản gốc: {document_title}
    Nội dung đoạn trích: {chunk_content}
    Yêu cầu: Viết đúng 1 câu duy nhất, bắt đầu bằng 'Đây là quy định về... nằm trong {document_title}...'.
    Không có lời bình luận nào khác.
    """
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Lỗi API Gemini khi xử lý file {document_title}: {e}")
        return ""

def process_and_save():
    logger.info(f"Bắt đầu quá trình Contextualize dữ liệu bằng Gemini (Model: {MODEL_ID})...")
    
    total_chunks = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        
        # Sắp xếp file theo tên để dễ theo dõi
        md_files = sorted([f for f in os.listdir(MD_FOLDER) if f.endswith('.md')])
        
        for filename in md_files:
            file_path = os.path.join(MD_FOLDER, filename)
            document_title = filename.replace(".md", "") 
            logger.info(f"Đang đọc file: {filename}")
            
            with open(file_path, "r", encoding="utf-8") as f_in:
                md_content = f_in.read()
            
            # Cắt Markdown thành các chunks
            md_chunks = markdown_splitter.split_text(md_content)
            logger.info(f"Đã cắt thành {len(md_chunks)} chunks.")
            
            for i, chunk in enumerate(md_chunks):
                original_text = chunk.page_content.strip()
                if not original_text:
                    continue
                    
                structure_metadata = chunk.metadata
                
                logger.info(f"Đang sinh ngữ cảnh cho chunk {i+1}/{len(md_chunks)}...")
                added_context = generate_context(original_text, document_title)
                
                # NGHỈ 4 GIÂY ĐỂ TRÁNH RATE LIMIT
                time.sleep(4) 
                
                contextualized_text = f"{added_context}\n\nNội dung chi tiết:\n{original_text}" if added_context else original_text
                
                # Đóng gói JSONL
                final_record = {
                    "chunk_id": f"{document_title}_chunk_{i+1}",
                    "metadata": {
                        "source_doc": document_title,
                        "hierarchy": structure_metadata
                    },
                    "original_text": original_text,
                    "contextualized_text": contextualized_text
                }
                
                f_out.write(json.dumps(final_record, ensure_ascii=False) + "\n")
                total_chunks += 1
                
    logger.info(f"Đã xử lý xong! Tổng cộng: {total_chunks} chunks.")
    logger.info(f"File lưu tại: {OUTPUT_FILE}")
    logger.info(f"Log được lưu tại: {log_filepath}")

if __name__ == "__main__":
    process_and_save()