"""
generator.py - Động cơ sinh câu trả lời pháp lý có căn cứ (Grounded Legal Reasoning).
Tích hợp LLM API cục bộ (Ollama / vLLM / OpenAI-compatible) và tuân thủ định dạng viện dẫn nghiêm ngặt.
"""

from __future__ import annotations

import os
import sys
from typing import List, Dict, Any, Optional

from openai import OpenAI

from configs.paths import ROOT_DIR
from configs.config import config
from configs.logging_config import get_subsystem_logger
from .retriever import LegalRetriever

logger = get_subsystem_logger("rag", "rag_engine")


class LegalGenerator:
    """Điều phối toàn bộ chu trình hỏi đáp: Truy xuất ngữ cảnh lai -> Xây dựng Prompt -> Sinh phản hồi."""

    def __init__(self, use_reranker: bool = False):
        logger.info("Khởi tạo LegalGenerator (Grounded Generation Layer)...")
        self.model = getattr(config, "GENERATOR_MODEL", "qwen2.5:1.5b")
        self.api_base = getattr(config, "LLM_API_BASE", "http://localhost:11434/v1")
        self.api_key = getattr(config, "LLM_API_KEY", "ollama")

        self.client = OpenAI(
            base_url=self.api_base,
            api_key=self.api_key,
            timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "45.0")),
            max_retries=1,
        )

        self.retriever = LegalRetriever(use_reranker=use_reranker)
        logger.info(f"✓ Generator sẵn sàng. Mô hình phục vụ: [{self.model}] tại {self.api_base}")

    def _call_llm(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.error(f"[LỖI LLM INFERENCE] {exc}")
            return f"Lỗi hệ thống: Không thể kết nối tới mô hình ngôn ngữ ({exc}). Vui lòng kiểm tra dịch vụ Ollama/vLLM."

    def build_prompt(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        context_blocks = []
        for i, c in enumerate(contexts, start=1):
            doc_name = c.get("source_doc") or c.get("doc_info") or f"Văn bản số {c.get('doc_number', 'N/A')}"
            eff_date = c.get("effective_date", "Chưa xác định")
            article = c.get("article", "Quy định liên quan")
            content = c.get("content", "").strip()

            block = (
                f"TÀI LIỆU {i}:\n"
                f"- Tên văn bản: {doc_name}\n"
                f"- Hiệu lực: {eff_date}\n"
                f"- Điều/Khoản: {article}\n"
                f"- Nội dung: {content}\n"
            )
            context_blocks.append(block)

        context_str = "\n".join(context_blocks)

        prompt = f"""Bạn là VietLawBERT, một chuyên gia tư vấn pháp luật tại Việt Nam.
NHIỆM VỤ: Dựa trên các tài liệu pháp luật đã được cung cấp trong phần CONTEXT, hãy trả lời câu hỏi của người dùng một cách chính xác và ngắn gọn nhất có thể.

CÁC QUY TẮC BẮT BUỘC:
1. Bắt đầu câu trả lời bằng cấu trúc: "Theo [Tên văn bản/Số hiệu] (có hiệu lực từ [Ngày hiệu lực]), [Nội dung Điều/Khoản]...", sau đó mới đi vào nội dung câu trả lời chi tiết.
2. TUYỆT ĐỐI không được bịa thêm thông tin ngoài phạm vi tài liệu đã cho.
3. Nếu context không chứa thông tin để trả lời, hãy thông báo rõ: "Dữ liệu hiện tại không đề cập đầy đủ thông tin để trả lời câu hỏi này."

CONTEXT:
{context_str}

CÂU HỎI: {query}
"""
        return prompt

    def ask(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        clean_query = query.strip()
        logger.info(f"Đang xử lý câu hỏi: '{clean_query[:80]}...'")

        contexts = self.retriever.search_context(clean_query, top_k=top_k)

        if not contexts:
            fallback_msg = "Dữ liệu hiện tại không đề cập, hệ thống không tìm thấy căn cứ pháp lý phù hợp để trả lời câu hỏi này."
            return {
                "query": clean_query,
                "answer": fallback_msg,
                "contexts": [],
                "has_context": False,
            }

        prompt = self.build_prompt(clean_query, contexts)
        answer = self._call_llm(prompt)

        return {
            "query": clean_query,
            "answer": answer,
            "contexts": contexts,
            "has_context": True,
        }

    def close(self):
        if hasattr(self.retriever, "close"):
            self.retriever.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Hệ thống hỏi đáp tư vấn pháp luật VietLawBERT RAG")
    parser.add_argument("--query", type=str, help="Câu hỏi pháp luật đầu vào")
    parser.add_argument("--top-k", type=int, default=3, help="Số lượng ngữ cảnh trích xuất")
    parser.add_argument("--rerank", action="store_true", help="Bật tầng Cross-Encoder Re-ranker")
    args = parser.parse_args()

    generator = LegalGenerator(use_reranker=args.rerank)
    test_query = args.query or "Vượt đèn đỏ đối với xe mô tô bị xử phạt bao nhiêu tiền?"

    print(f"\n=======================================================")
    print(f"CÂU HỎI: {test_query}")
    print(f"=======================================================")

    result = generator.ask(test_query, top_k=args.top_k)

    print("\n--- TRẢ LỜI TỪ VIETLAWBERT ---")
    print(result["answer"])

    print("\n--- CĂN CỨ TRUY XUẤT (TOP CONTEXTS) ---")
    for idx, ctx in enumerate(result["contexts"], start=1):
        print(f"[{idx}] {ctx.get('article')} | {ctx.get('doc_number')} (RRF Score: {ctx.get('rrf_score')})")

    generator.close()


if __name__ == "__main__":
    main()