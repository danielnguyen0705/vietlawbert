import os
import sys
import json
import time
import logging
from datetime import datetime
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from google import genai
from google.genai import types

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN & LOGGING ĐỘNG
# ==========================================

# Tự động xác định BASE_DIR (law_dataset/)
# Giả sử file nằm ở: law_dataset/src/preprocess/contextualizer.py
# Cấp 1: preprocess/, Cấp 2: src/, Cấp 3: law_dataset/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../../"))

# Cấu hình thư mục dữ liệu
MD_FOLDER = os.path.join(BASE_DIR, "data", "processed")
JSON_DIR = os.path.join(BASE_DIR, "json")
os.makedirs(JSON_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(JSON_DIR, "final_contextual_chunks.jsonl")
METADATA_FILE = os.path.join(JSON_DIR, "metadata.jsonl")

# Cấu hình Thư mục Log theo ngày (Y-m-d)
TODAY_STR = datetime.now().strftime("%Y-%m-%d")
LOG_DIR = os.path.join(BASE_DIR, "data", "logs", TODAY_STR)
os.makedirs(LOG_DIR, exist_ok=True)

# Tự động lấy tên file (contextualizer) làm tên log
CURRENT_FILENAME = os.path.basename(__file__).split('.')[0]
LOG_FILE_PATH = os.path.join(LOG_DIR, f"log_{CURRENT_FILENAME}.log")

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        # mode="a" đảm bảo chạy nhiều lần trong ngày sẽ append nối tiếp vào file cũ
        logging.FileHandler(LOG_FILE_PATH, encoding="utf-8", mode="a"),
        logging.StreamHandler(sys.stdout)
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
        logger.error("Không tìm thấy GEMINI_API_KEY.")
        raise ValueError("Thiếu API Key.")

client = genai.Client(api_key=api_key)
MODEL_ID = 'gemini-2.5-flash' 

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

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,     # Chiều dài lý tưởng cho LLM đọc
    chunk_overlap=150,   # Giữ lại 150 ký tự gối đầu để không đứt ngữ nghĩa
    separators=["\n\n", "\n", ".", " ", ""]
)

def generate_context(chunk_content, document_title):
    prompt = f"""
    Bạn là một chuyên gia pháp lý. Hãy viết 1 câu ngữ cảnh giải thích ngắn gọn cho đoạn văn bản luật sau.
    Tên/Số hiệu văn bản: {document_title}
    Nội dung đoạn trích: {chunk_content}
    Yêu cầu: Viết đúng 1 câu duy nhất, bắt đầu bằng 'Đây là quy định về... nằm trong {document_title}...'.
    Không có lời bình luận nào khác.
    """
    try:
        response = client.models.generate_content(model=MODEL_ID, contents=prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Lỗi API Gemini: {e}")
        return ""

def process_and_save():
    logger.info(f"Bắt đầu quá trình Contextualize dữ liệu bằng Gemini...")
    
    # ĐỌC METADATA ĐỂ LÀM TỪ ĐIỂN (LOOKUP)
    meta_lookup = {}
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    # Lấy doc_number và effFrom (Ngày hiệu lực)
                    meta_lookup[data['item_id']] = {
                        "doc_number": data.get("doc_number", data['item_id']),
                        "effFrom": data.get("metadata_api", {}).get("effFrom", "Chưa xác định")
                    }
        logger.info(f"Đã tải thành công {len(meta_lookup)} records từ metadata.jsonl")
    else:
        logger.warning("Không tìm thấy metadata.jsonl. Dữ liệu sẽ thiếu doc_number và ngày hiệu lực!")

    total_chunks = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        md_files = sorted([f for f in os.listdir(MD_FOLDER) if f.endswith('.md')])
        
        for filename in md_files:
            file_path = os.path.join(MD_FOLDER, filename)
            doc_id = filename.replace(".md", "") 
            
            # Tra cứu lấy Số hiệu và Ngày hiệu lực thật
            doc_meta = meta_lookup.get(doc_id, {"doc_number": doc_id, "effFrom": "Chưa xác định"})
            real_doc_number = doc_meta["doc_number"]
            real_eff_date = doc_meta["effFrom"]
            
            logger.info(f"Đang xử lý: {real_doc_number} (ID: {doc_id})")
            
            with open(file_path, "r", encoding="utf-8") as f_in:
                md_content = f_in.read()
            
            md_chunks = markdown_splitter.split_text(md_content)
            
            final_chunks = []
            for chunk in md_chunks:
                if len(chunk.page_content) > 1500:
                    # Nếu băm theo Điều/Khoản rồi mà vẫn quá dài -> Băm nhỏ tiếp theo đoạn văn
                    sub_texts = text_splitter.split_text(chunk.page_content)
                    for sub in sub_texts:
                        final_chunks.append({"text": sub, "metadata": chunk.metadata})
                else:
                    final_chunks.append({"text": chunk.page_content, "metadata": chunk.metadata})
            
            # Thay vì lặp md_chunks, giờ ta lặp final_chunks đã được bảo vệ
            for i, chunk_data in enumerate(final_chunks):
                original_text = chunk_data["text"].strip()
                if not original_text:
                    continue
                    
                structure_metadata = chunk_data["metadata"]
                
                # Gọi API với Số hiệu thật để AI hiểu rõ ngữ cảnh hơn
                added_context = generate_context(original_text, real_doc_number)
                time.sleep(4) # Chống Rate Limit
                
                contextualized_text = f"{added_context}\n\nNội dung chi tiết:\n{original_text}" if added_context else original_text
                
                # ĐÓNG GÓI JSONL VỚI ĐẦY ĐỦ METADATA CẦN THIẾT
                final_record = {
                    "chunk_id": f"{doc_id}_chunk_{i+1}",
                    "metadata": {
                        "doc_id": doc_id,
                        "doc_number": real_doc_number,
                        "effective_date": real_eff_date, # Đã chuẩn hóa đúng key
                        "hierarchy": structure_metadata
                    },
                    "original_text": original_text,
                    "contextualized_text": contextualized_text
                }
                
                f_out.write(json.dumps(final_record, ensure_ascii=False) + "\n")
                total_chunks += 1
                
    logger.info(f"Hoàn tất! Đã xử lý {total_chunks} chunks.")
    logger.info(f"File lưu tại: {OUTPUT_FILE}")
    logger.info(f"Log được lưu tại: {log_filepath}")

if __name__ == "__main__":
    process_and_save()