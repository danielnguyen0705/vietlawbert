"""
generator.py - Sinh câu trả lời (Layer 4: Online Reasoning)

Dùng Ollama deepseek-r1:14b thay Gemini.
"""

import os
import sys
import logging
import requests
from datetime import datetime
from dotenv import load_dotenv

from paths import BASE_DIR, get_log_path
from rag.retriever import LegalRetriever

load_dotenv(os.path.join(BASE_DIR, ".env"))

from openai import OpenAI
from config import config

CURRENT_FILENAME = os.path.basename(__file__).split('.')[0]
LOG_FILE_PATH = get_log_path(CURRENT_FILENAME)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(levelname)s] | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, encoding="utf-8", mode="a"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(CURRENT_FILENAME.capitalize())


class LegalGenerator:
    def __init__(self):
        logger.info("Khoi tao Generator (OpenAI standard API)...")
        self.model = config.GENERATOR_MODEL
        self.client = OpenAI(base_url=config.LLM_API_BASE, api_key=config.LLM_API_KEY)

        self.retriever = LegalRetriever()
        logger.info("Generator da san sang!")

    def _call_ollama(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"[LLM ERROR] {e}")
            return f"Loi goi LLM: {e}"

    def ask(self, query: str) -> str:
        print(f"\n{'='*50}\nCAU HOI: {query}\n{'='*50}")

        contexts = self.retriever.search_context(query)

        if not contexts:
            return "Dữ liệu hiện tại không đề cập, tôi chỉ chuyên về luật giao thông và không thể trả lời câu hỏi đó."

        context_str = ""
        for i, c in enumerate(contexts):
            context_str += f"TAI LIEU {i+1}:\n- Van ban: {c['doc_info']}\n- Hieu luc: {c['effective_date']}\n- Dieu: {c['article']}\n- Noi dung: {c['content']}\n\n"

        prompt = f"""
Ban la VietLawBERT, mot chuyen gia ve luat giao thong tai Viet Nam.
NHIEM VU: Dựa trên cac tai lieu luat giao thong da duoc cung cap trong phan CONTEXT, hay tra loi cau hoi cua nguoi dung mot cach chinh xac va ngan gon nhat co the.

CAC QUY TAC BAT BUOC:
1. Bat dau cau tra loi bang cau truc: "Theo [Ten van ban/So hieu] (co hieu luc tu [Ngay hieu luc]), ...", Sau do moi di vao noi dung cau tra loi ngan gon.
2. TUYET DOI khong duoc bia them thong tin.
3. Neu context khong co thong tin, hay bao la "Du lieu hien tai khong de cap, toi chi chuyen ve luat giao thong va khong the tra loi cau hoi do".

CONTEXT:
{context_str}

CAU HOI: {query}
"""

        logger.info(f"Dang xu ly cau hoi: {query[:60]}...")
        response = self._call_ollama(prompt)

        print(f"TRA LOI TU VIETLAWBERT:\n{response}")
        logger.info(f"Cau tra loi cho '{query[:60]}': {response[:200]}...")
        return response


if __name__ == "__main__":
    pass