"""
generator.py - Động cơ sinh câu trả lời pháp lý với cơ chế Cascading Fallback (Kiến trúc v3).
Tự động luân chuyển mô hình theo độ ưu tiên: Cloud SOTA -> Cloud Fast -> Local vLLM/Ollama.
Tích hợp tính toán Attribution Score phục vụ đánh giá RQ4 (Grounded Legal Reasoning).
"""

from __future__ import annotations

import os
import re
import logging
import unicodedata
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from openai import OpenAI

from configs.config import config
from configs.logging_config import get_subsystem_logger
from .retriever import LegalHybridRetriever

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
    timeout: float = 30.0


class ResilientLLMDispatcher:
    """Điều phối suy luận với cơ chế Fallback tuần tự theo thứ tự ưu tiên."""

    def __init__(self):
        self.cascade_chain: List[LLMProviderNode] = self._build_cascade_chain()
        self._clients: Dict[str, OpenAI] = {}

    def _build_cascade_chain(self) -> List[LLMProviderNode]:
        chain = []

        cloud_key = os.getenv("PRIMARY_LLM_API_KEY") or getattr(config, "LLM_API_KEY", "")
        cloud_base = os.getenv("PRIMARY_LLM_API_BASE", "https://api.openai.com/v1")
        primary_model = os.getenv("PRIMARY_LLM_MODEL", "gpt-4o")

        if cloud_key and cloud_key.lower() != "ollama":
            chain.append(LLMProviderNode(
                name="Primary_Cloud_SOTA",
                model_id=primary_model,
                api_base=cloud_base,
                api_key=cloud_key,
                timeout=float(os.getenv("PRIMARY_LLM_TIMEOUT_SECONDS", "45")),
            ))

        fallback_key = os.getenv("FALLBACK_LLM_API_KEY")
        fallback_base = os.getenv("FALLBACK_LLM_API_BASE", "https://generativelanguage.googleapis.com/v1beta/openai/")
        fallback_model = os.getenv("FALLBACK_LLM_MODEL", "gemini-1.5-flash")

        if fallback_key:
            chain.append(LLMProviderNode(
                name="Secondary_Cloud_Fast",
                model_id=fallback_model,
                api_base=fallback_base,
                api_key=fallback_key,
                timeout=20.0,
            ))

        local_base = getattr(config, "LLM_API_BASE", "http://localhost:11434/v1")
        local_model = getattr(config, "GENERATOR_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        local_key = getattr(config, "LLM_API_KEY", "ollama")

        if os.getenv("ENABLE_LOCAL_LLM", "true").lower() == "true":
            chain.append(LLMProviderNode(
                name="Local_Safety_Net",
                model_id=local_model,
                api_base=local_base,
                api_key=local_key,
                timeout=60.0,
            ))

        return chain

    def _get_client(self, node: LLMProviderNode) -> OpenAI:
        if node.name not in self._clients:
            self._clients[node.name] = OpenAI(
                base_url=node.api_base,
                api_key=node.api_key,
                timeout=node.timeout,
                max_retries=1,
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
                answer = (response.choices[0].message.content or "").strip()
                if answer:
                    logger.info("Mô hình [%s] phản hồi thành công.", node.name)
                    return answer, node.name
            except Exception as exc:
                logger.warning("Mô hình [%s] không phản hồi (%s). Đang chuyển sang tầng kế tiếp...", node.name, exc)
                last_exception = exc
                continue

        logger.error("Toàn bộ chuỗi Cascade đều thất bại. Lỗi cuối cùng: %s", last_exception)
        return "Lỗi hệ thống: Hiện không thể kết nối tới bất kỳ dịch vụ AI nào để tổng hợp câu trả lời.", "Failed"


class LegalGenerator:
    """Tầng tạo sinh câu trả lời tích hợp kiểm chứng căn cứ và Cascade Fallback tự động."""

    def __init__(
        self,
        retriever: Optional[LegalHybridRetriever] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        self.retriever = retriever or LegalHybridRetriever()
        self.dispatcher = ResilientLLMDispatcher()
        self.temperature = temperature if temperature is not None else float(os.getenv("LLM_TEMPERATURE", "0.1"))
        self.max_tokens = max_tokens if max_tokens is not None else int(os.getenv("LLM_MAX_TOKENS", "1024"))
        logger.info("LegalGenerator khởi tạo thành công với chuỗi điều phối %d tầng.", len(self.dispatcher.cascade_chain))

    def _format_context(self, contexts: List[Dict[str, Any]]) -> str:
        context_blocks = []
        for i, c in enumerate(contexts, start=1):
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

    def _compute_attribution_score(self, answer: str, contexts: List[Dict[str, Any]]) -> float:
        """Đo lường định lượng tỷ lệ căn cứ xuất hiện trong câu trả lời phục vụ RQ4 (Faithfulness)."""
        if not contexts:
            return 0.0

        def normalize_citation(value: Any) -> str:
            text = unicodedata.normalize("NFKC", str(value or "")).casefold()
            text = re.sub(r"[‐‑‒–—―−]", "-", text)
            return re.sub(r"\s+", " ", text).strip()

        matched = 0
        normalized_answer = normalize_citation(answer)
        for c in contexts:
            doc_num = normalize_citation(c.get("doc_number"))
            h_path = normalize_citation(c.get("hierarchy_path"))

            art_match = re.search(r"điều\s+(\d+[a-z]?)", h_path)
            art_str = art_match.group(0) if art_match else ""

            # Lookarounds avoid matching Điều 1 inside Điều 10/12 while still
            # supporting Vietnamese letters and punctuation-heavy document IDs.
            has_doc = bool(doc_num and doc_num != "n/a" and re.search(
                r"(?<!\w)" + re.escape(doc_num) + r"(?!\w)", normalized_answer,
            ))
            has_art = bool(art_str and re.search(
                r"(?<!\w)" + re.escape(art_str) + r"(?!\w)", normalized_answer,
            ))

            if has_doc or has_art:
                matched += 1

        return round(matched / len(contexts), 4)

    def ask(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        clean_query = query.strip()
        logger.info("Đang xử lý câu hỏi: '%s...'", clean_query[:80])

        contexts = self.retriever.retrieve(clean_query, top_k=top_k)
        if not contexts:
            fallback_msg = "Cơ sở dữ liệu hiện tại không đủ căn cứ pháp lý để giải đáp thắc mắc này."
            return {
                "query": clean_query,
                "answer": fallback_msg,
                "contexts": [],
                "has_context": False,
                "attribution_score": 0.0,
                "model_used": "None",
            }

        context_str = self._format_context(contexts)
        prompt = LEGAL_PROMPT_TEMPLATE.format(context_blocks=context_str, query=clean_query)
        answer, provider_used = self.dispatcher.generate(
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        attr_score = self._compute_attribution_score(answer, contexts)

        return {
            "query": clean_query,
            "answer": answer,
            "contexts": contexts,
            "has_context": True,
            "attribution_score": attr_score,
            "model_used": provider_used,
        }

    def generate_response(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        return self.ask(query=query, top_k=top_k)

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

    print("\n" + "=" * 55)
    print(f"CÂU HỎI: {test_query}")
    print("=" * 55)

    result = generator.ask(test_query, top_k=args.top_k)

    print("\n--- TRẢ LỜI TỪ VIETLAWBERT ---")
    print(result["answer"])
    print(f"\n[Model Used: {result['model_used']} | Attribution Score (RQ4): {result['attribution_score']}]")

    print("\n--- CĂN CỨ TRUY XUẤT (TOP CONTEXTS) ---")
    for idx, ctx in enumerate(result["contexts"], start=1):
        print(f"[{idx}] {ctx.get('hierarchy_path')} | {ctx.get('doc_number')} (Score: {ctx.get('final_rerank_score')})")

    generator.close()


if __name__ == "__main__":
    main()
