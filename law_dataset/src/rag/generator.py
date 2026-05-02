import os
import sys
import logging
from google import genai
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from retriever import LegalRetriever 
except ImportError:
    from .retriever import LegalRetriever 

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN & LOGGING TỰ ĐỘNG
# ==========================================
current_file_name = os.path.splitext(os.path.basename(__file__))[0]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(BASE_DIR, ".env"))

LOG_DIR = os.path.join(BASE_DIR, "data", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

log_filepath = os.path.join(LOG_DIR, f"log_{current_file_name}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filepath, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(current_file_name.capitalize())

class LegalGenerator:
    def __init__(self):
        logger.info("🧠 Đang khởi tạo Generator (Gemini SDK mới)...")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("❌ Thiếu GEMINI_API_KEY trong file .env!")
            sys.exit(1)
            
        self.client = genai.Client(api_key=api_key) 
        self.model_id = "gemini-2.5-flash"
        
        self.retriever = LegalRetriever()
        logger.info("✅ Generator đã sẵn sàng!")

    def ask(self, query: str):
        print(f"\n{'='*50}\n🙋‍♂️ CÂU HỎI: {query}\n{'='*50}")
        
        contexts = self.retriever.search_context(query)
        
        # Đưa thông tin chi tiết vào Prompt
        context_str = ""
        for i, c in enumerate(contexts):
            context_str += f"TÀI LIỆU {i+1}:\n- Văn bản: {c['doc_info']}\n- Hiệu lực: {c['effective_date']}\n- Điều: {c['article']}\n- Nội dung: {c['content']}\n\n"

        prompt = f"""
        Bạn là VietLawBERT, một chuyên gia về luật giao thông tại Việt Nam. 
        NHIỆM VỤ: Dựa trên các tài liệu luật giao thông đã được cung cấp trong phần CONTEXT, hãy trả lời câu hỏi của người dùng một cách chính xác và ngắn gọn nhất có thể.

        CÁC QUY TẮC BẮT BUỘC:
        1. Bắt đầu câu trả lời bằng cấu trúc: "Theo [Tên văn bản/Số hiệu] (có hiệu lực từ [Ngày hiệu lực]), ...", Sau đó mới đi vào nội dung câu trả lời ngắn gọn.
        2. TUYỆT ĐỐI không được bịa thêm thông tin.
        3. Nếu context không có thông tin, hãy báo là "Dữ liệu hiện tại không đề cập, tôi chỉ chuyên về luật giao thông và không thể trả lời câu hỏi đó".

        CONTEXT:
        {context_str}

        CÂU HỎI: {query}
        """

        logger.info("🤖 Gemini đang xử lý câu trả lời...")
        
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt
        )
        
        print(f"TRẢ LỜI TỪ VIETLAWBERT:\n{response.text}")
        logger.info(f"Câu trả lời cho '{query}':\\n{response.text}")
        return response.text

# Đã xóa bot.ask() ở đây để file này chỉ làm nhiệm vụ cung cấp "não" cho app.py
if __name__ == "__main__":
    pass