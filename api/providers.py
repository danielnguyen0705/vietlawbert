"""Providers share ask/close; the real engine is imported only on selection."""
from api.schemas import LawContext

DEMO_TOPICS = {
    "giao thông": ("GT", "Quy tắc giao thông minh họa"),
    "doanh nghiệp": ("DN", "Thủ tục doanh nghiệp minh họa"),
    "hành chính": ("HC", "Quy trình hành chính minh họa"),
}


class ProviderFailure(Exception):
    pass


class MockProvider:
    def ask(self, query: str, top_k: int):
        match = next((value for key, value in DEMO_TOPICS.items() if key in query.casefold()), None)
        contexts = []
        if match:
            code, title = match
            contexts = [LawContext(
                chunk_id=f"demo-{code}-{i}", doc_number=f"DEMO-{code}-{i}",
                hierarchy_path=f"Mục minh họa {i}",
                content=f"DỮ LIỆU GIẢ LẬP — {title}, mẫu {i}. Nội dung này chỉ kiểm tra cách hiển thị căn cứ, không phải quy định pháp luật.",
                final_rerank_score=round(0.95 - i * 0.05, 2), graph_boost=0.1,
            ).model_dump() for i in range(1, min(top_k, 3) + 1)]
        answer = (
            "**Chế độ demo — dữ liệu minh họa.**\n\n"
            + ("Đã nhận câu hỏi thuộc chủ đề minh họa. Các căn cứ "
               + ", ".join(c["doc_number"] for c in contexts)
               + " bên dưới là dữ liệu giả lập để kiểm tra giao diện, không dùng làm tư vấn pháp lý."
               if contexts else "Chưa có căn cứ demo cho câu hỏi này. Hãy thử chủ đề giao thông, doanh nghiệp hoặc hành chính.")
        )
        return dict(query=query, answer=answer, contexts=contexts, has_context=bool(contexts),
                    attribution_score=1.0 if contexts else 0.0, model_used="mock-fixture")

    def close(self):
        pass


class RagProvider:
    def __init__(self, generator=None):
        if generator is None:
            from rag.generator import LegalGenerator
            generator = LegalGenerator()
        self.generator = generator

    def ask(self, query: str, top_k: int):
        result = self.generator.ask(query=query, top_k=top_k)
        if result.get("model_used") == "Failed":
            raise ProviderFailure("generation_failed")
        contexts = []
        for raw in result.get("contexts", [])[:top_k]:
            context = {key: raw.get(key) for key in LawContext.model_fields}
            context["content"] = raw.get("content") or raw.get("text") or ""
            contexts.append(LawContext.model_validate(context).model_dump())
        return dict(query=query, answer=result["answer"], contexts=contexts,
                    has_context=bool(contexts), attribution_score=result.get("attribution_score"),
                    model_used=result.get("model_used"))

    def close(self):
        self.generator.close()


def make_provider(mode: str):
    return MockProvider() if mode == "mock" else RagProvider()
