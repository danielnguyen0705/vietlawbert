"""
generator.py - Động cơ sinh câu trả lời pháp lý với cơ chế Cascading Fallback (Kiến trúc v3).
Tự động luân chuyển mô hình theo độ ưu tiên: Cloud SOTA (Claude/OpenRouter) -> Cloud Fast (Gemini) -> Local vLLM/Ollama.
Tích hợp tính toán Attribution Score phục vụ đánh giá RQ4 (Grounded Legal Reasoning).
"""

from __future__ import annotations

import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import httpx
from openai import OpenAI

from configs.config import config
from configs.logging_config import get_subsystem_logger
from rag.retriever import LegalHybridRetriever

logger = get_subsystem_logger("rag", "generator")

LEGAL_PROMPT_TEMPLATE = """Bạn là trợ lý ảo pháp lý chuyên gia dựa trên hệ thống VietLawBERT.
Nhiệm vụ của bạn là trả lời câu hỏi của người dùng DỰA HOÀN TOÀN vào các căn cứ pháp lý được cung cấp dưới đây.

QUY TẮC BẮT BUỘC:
1. Mở đầu bằng cấu trúc: "Căn cứ theo [Tên/Số hiệu văn bản], tại [Điều/Khoản] quy định: ..." sau đó mới trình bày nội dung giải thích chi tiết.
2. Tuyệt đối không suy diễn hoặc đưa thông tin ngoài phạm vi tài liệu đã cho.
3. Nếu các tài liệu không chứa đầy đủ thông tin để trả lời, thông báo rõ: "Cơ sở dữ liệu hiện tại không đủ căn cứ pháp lý để giải đáp thắc mắc này."

=== CĂN CỨ PHÁP LÝ TRUY XUẤT ===
{context_blocks}

=== CÂU HỎI CỦA NGƯỜI DÙNG ===
{query}

=== CÂU TRẢ LỜI PHÁP LÝ ===
"""


@dataclass
class LLMProviderNode:
    name: str
    model_id: str
    api_base: str
    api_key: str
    timeout: float = 20.0


class ResilientLLMDispatcher:
    """Điều phối suy luận với cơ chế Fallback tuần tự theo thứ tự ưu tiên."""

    def __init__(self):
        self.cascade_chain: List[LLMProviderNode] = self._build_cascade_chain()
        self._clients: Dict[str, OpenAI] = {}

    def _build_cascade_chain(self) -> List[LLMProviderNode]:
        chain = []

        # Tầng 1: Primary Cloud SOTA (Claude 3.5 Sonnet qua OpenRouter / Cloud API)
        # Thiết lập timeout 15.0s: nếu proxy mạng treo quá 15s lập tức failover sang tầng kế tiếp
        cloud_key = os.getenv("PRIMARY_LLM_API_KEY") or getattr(config, "PRIMARY_LLM_API_KEY", "")
        cloud_base = os.getenv("PRIMARY_LLM_API_BASE") or getattr(config, "PRIMARY_LLM_API_BASE", "https://openrouter.ai/api/v1")
        primary_model = os.getenv("PRIMARY_LLM_MODEL") or getattr(config, "PRIMARY_LLM_MODEL", "claude-3.5-sonnet")

        if cloud_key and cloud_key.strip() and cloud_key.lower() != "ollama":
            chain.append(LLMProviderNode(
                name="Primary_Cloud_SOTA",
                model_id=primary_model,
                api_base=cloud_base,
                api_key=cloud_key,
                timeout=15.0,
            ))

        # Tầng 2: Secondary Cloud Fast (Gemini Flash / Proxy)
        # Tốc độ phản hồi cực nhanh (~5s - 8s), đóng vai trò cứu cánh tức thời khi SOTA nghẽn
        fallback_key = os.getenv("FALLBACK_LLM_API_KEY") or getattr(config, "FALLBACK_LLM_API_KEY", "")
        fallback_base = os.getenv("FALLBACK_LLM_API_BASE") or getattr(config, "FALLBACK_LLM_API_BASE", "https://generativelanguage.googleapis.com/v1beta/openai/")
        fallback_model = os.getenv("FALLBACK_LLM_MODEL") or getattr(config, "FALLBACK_LLM_MODEL", "gemini-3.8-flash")

        if fallback_key and fallback_key.strip():
            chain.append(LLMProviderNode(
                name="Secondary_Cloud_Fast",
                model_id=fallback_model,
                api_base=fallback_base,
                api_key=fallback_key,
                timeout=20.0,
            ))

        # Tầng 3: Local Safety Net (vLLM / Ollama trên máy trạm - Luôn sẵn sàng)
        local_base = os.getenv("LLM_API_BASE") or getattr(config, "LLM_API_BASE", "http://localhost:11434/v1")
        local_model = os.getenv("GENERATOR_MODEL") or getattr(config, "GENERATOR_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        local_key = os.getenv("LLM_API_KEY") or getattr(config, "LLM_API_KEY", "ollama")

        chain.append(LLMProviderNode(
            name="Local_Safety_Net",
            model_id=local_model,
            api_base=local_base,
            api_key=local_key,
            timeout=30.0,
        ))

        return chain

    def _get_client(self, node: LLMProviderNode) -> OpenAI:
        if node.name not in self._clients:
            # Sử dụng httpx.Timeout chặt chẽ: connect tối đa 5s, total tối đa node.timeout
            # Khóa max_retries=0 để lập tức chuyển tiếp (failover) sang node kế tiếp trong chuỗi Cascade khi lỗi
            self._clients[node.name] = OpenAI(
                base_url=node.api_base,
                api_key=node.api_key,
                timeout=httpx.Timeout(node.timeout, connect=5.0),
                max_retries=0,
            )
        return self._clients[node.name]

    def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 1024) -> Tuple[str, str]:
        last_exception = None

        for node in self.cascade_chain:
            try:
                logger.info("Đang điều phối suy luận tới [%s] (%s)...", node.name, node.model_id)
                client = self._get_client(node)
                response = client.chat.completions.create(
                    model=node.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                answer = response.choices[0].message.content.strip()
                if answer:
                    logger.info("Mô hình [%s] phản hồi thành công.", node.name)
                    return answer, node.name
            except Exception as exc:
                logger.warning("Mô hình [%s] không phản hồi (%s). Chuyển sang tầng kế tiếp...", node.name, exc)
                last_exception = exc
                continue

        logger.error("Toàn bộ chuỗi Cascade đều thất bại. Lỗi cuối cùng: %s", last_exception)
        return "Lỗi hệ thống: Hiện không thể kết nối tới bất kỳ dịch vụ AI nào để tổng hợp câu trả lời.", "Failed"


class LegalGenerator:
    """Tầng tạo sinh câu trả lời tích hợp kiểm chứng căn cứ và Cascade Fallback tự động."""

    def __init__(
        self,
        retriever: Optional[LegalHybridRetriever] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ):
        self.retriever = retriever or LegalHybridRetriever()
        self.dispatcher = ResilientLLMDispatcher()
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.info("LegalGenerator khởi tạo thành công với chuỗi điều phối %d tầng.", len(self.dispatcher.cascade_chain))

    def _format_context(self, contexts: List[Any]) -> str:
        context_blocks = []
        for i, c in enumerate(contexts, start=1):
            if isinstance(c, str):
                # Hỗ trợ trường hợp contexts đã được định dạng sẵn chuỗi từ provider
                context_blocks.append(f"TÀI LIỆU [{i}]:\n{c}")
                continue

            doc_num = c.get("doc_number") or c.get("doc_id") or "N/A"
            eff_date = c.get("effective_date", "Chưa xác định")
            h_path = c.get("hierarchy_path") or c.get("article") or "Điều khoản liên quan"
            content = (c.get("content") or c.get("text") or "").strip()

            block = (
                f"TÀI LIỆU [{i}]:\n"
                f"- Số hiệu: {doc_num}\n"
                f"- Hiệu lực: {eff_date}\n"
                f"- Phân vị: {h_path}\n"
                f"- Nội dung: {content}"
            )
            context_blocks.append(block)

        return "\n\n".join(context_blocks)

    def _compute_attribution_score(self, answer: str, contexts: List[Any]) -> float:
        """Đo lường định lượng tỷ lệ căn cứ xuất hiện trong câu trả lời phục vụ RQ4 (Faithfulness)."""
        if not contexts:
            return 0.0
        matched = 0
        ans_lower = answer.lower()
        for c in contexts:
            if isinstance(c, str):
                doc_matches = re.findall(r"\d+/\d+/[A-ZĐCPTtgHĐND-]+", c, re.IGNORECASE)
                art_match = re.search(r"điều\s+(\d+[a-za-z]?)", c, re.IGNORECASE)
                has_doc = any(dm.lower() in ans_lower for dm in doc_matches)
                has_art = bool(art_match and art_match.group(0).lower() in ans_lower)
            else:
                doc_num = str(c.get("doc_number", "")).strip().lower()
                h_path = str(c.get("hierarchy_path", "")).strip().lower()

                art_match = re.search(r"điều\s+(\d+[a-za-z]?)", h_path)
                art_str = art_match.group(0) if art_match else ""

                has_doc = bool(doc_num and doc_num != "n/a" and doc_num in ans_lower)
                has_art = bool(art_str and art_str in ans_lower)

            if has_doc or has_art:
                matched += 1

        return round(matched / len(contexts), 4)

    def ask(self, query: str, top_k: int = 3, contexts: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Thực hiện sinh câu trả lời pháp lý dựa trên căn cứ.
        Nếu contexts đã được truyền vào từ bên ngoài (ví dụ generation_provider), tái sử dụng trực tiếp để tránh truy xuất 2 lần.
        """
        clean_query = query.strip()
        logger.info("Đang xử lý câu hỏi: '%s...'", clean_query[:80])

        if contexts is None:
            retrieved_contexts = self.retriever.retrieve(clean_query, top_k=top_k)
        else:
            retrieved_contexts = contexts

        if not retrieved_contexts:
            fallback_msg = "Cơ sở dữ liệu hiện tại không đủ căn cứ pháp lý để giải đáp thắc mắc này."
            return {
                "query": clean_query,
                "answer": fallback_msg,
                "contexts": [],
                "has_context": False,
                "attribution_score": 0.0,
                "model_used": "None",
            }

        context_str = self._format_context(retrieved_contexts)
        prompt = LEGAL_PROMPT_TEMPLATE.format(context_blocks=context_str, query=clean_query)
        answer, provider_used = self.dispatcher.generate(
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        attr_score = self._compute_attribution_score(answer, retrieved_contexts)

        return {
            "query": clean_query,
            "answer": answer,
            "contexts": retrieved_contexts,
            "has_context": True,
            "attribution_score": attr_score,
            "model_used": provider_used,
        }

    def generate_response(self, query: str, top_k: int = 3, contexts: Optional[List[Any]] = None) -> Dict[str, Any]:
        return self.ask(query=query, top_k=top_k, contexts=contexts)

    def close(self) -> None:
        if hasattr(self.retriever, "close"):
            self.retriever.close()


LegalAnswerGenerator = LegalGenerator


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Hệ thống hỏi đáp tư vấn pháp luật VietLawBERT RAG")
    parser.add_argument("--query", type=str, help="Câu hỏi pháp luật đầu vào")
    parser.add_argument("--top-k", type=int, default=3, help="Số lượng ngữ cảnh trích xuất")
    args = parser.parse_args()

    generator = LegalGenerator()
    test_query = args.query or "Vượt đèn đỏ đối với xe mô tô bị xử phạt bao nhiêu tiền?"

    print("\n" + "=" * 60)
    print(f"CÂU HỎI: {test_query}")
    print("=" * 60)

    result = generator.ask(test_query, top_k=args.top_k)

    print("\n--- TRẢ LỜI TỪ VIETLAWBERT ---")
    print(result["answer"])
    print(f"\n[Model Used: {result['model_used']} | Attribution Score (RQ4): {result['attribution_score']}]")

    print("\n--- CĂN CỨ TRUY XUẤT (TOP CONTEXTS) ---")
    for idx, ctx in enumerate(result["contexts"], start=1):
        if isinstance(ctx, dict):
            print(f"[{idx}] {ctx.get('hierarchy_path')} | {ctx.get('doc_number')} (Score: {ctx.get('final_rerank_score')})")
        else:
            print(f"[{idx}] {str(ctx)[:120]}...")

    generator.close()


if __name__ == "__main__":
    main()