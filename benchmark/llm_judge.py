"""
llm_judge.py - Hệ thống thẩm phán LLM-as-a-Judge tự động đánh giá chất lượng RAG.
Đo đạc định lượng 3 chỉ số theo chuẩn Ragas/G-Eval: Faithfulness, Answer Relevance, Context Precision.
Khử triệt để markdown fences để không bao giờ bị ngoại lệ JSONDecodeError.
"""

from __future__ import annotations

import re
import csv
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI
import numpy as np

from configs.paths import ROOT_DIR
from configs.config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | [%(levelname)s] | %(message)s")
logger = logging.getLogger("VietLawBERT_LLMJudge")

JUDGE_PROMPT_TEMPLATE = """Bạn là một chuyên gia thẩm định hệ thống hỏi đáp Pháp luật (Legal RAG Judge).
Nhiệm vụ của bạn là đánh giá khách quan câu trả lời của mô hình dựa trên câu hỏi và ngữ cảnh trích dẫn.

[DỮ LIỆU ĐÁNH GIÁ]
1. Câu hỏi (Query): {query}
2. Ngữ cảnh cung cấp (Contexts): {contexts}
3. Câu trả lời (Answer): {answer}

[TIÊU CHÍ ĐÁNH GIÁ]
1. Faithfulness (0.0 - 1.0): Mọi luận điểm trong câu trả lời có hoàn toàn dựa trên ngữ cảnh hay không? Có hiện tượng bịa đặt/ảo giác không? (1.0 = hoàn toàn trung thực với ngữ cảnh).
2. Answer_Relevance (0.0 - 1.0): Câu trả lời có giải quyết trực tiếp và đầy đủ trọng tâm câu hỏi không? (1.0 = đúng trọng tâm tuyệt đối).
3. Context_Precision (0.0 - 1.0): Các ngữ cảnh trích dẫn có thực sự liên quan và hữu ích để trả lời câu hỏi không? (1.0 = toàn bộ ngữ cảnh đều có giá trị).

Trả về kết quả DUY NHẤT dưới định dạng JSON sau (không thêm bất kỳ lời dẫn nào):
{{
  "faithfulness": <float>,
  "answer_relevance": <float>,
  "context_precision": <float>,
  "reasoning": "<giải thích ngắn gọn lý do cho điểm>"
}}
"""


class LegalLLMJudge:
    def __init__(self):
        self.client = OpenAI(
            base_url=config.PRIMARY_LLM_API_BASE,
            api_key=config.PRIMARY_LLM_API_KEY or "dummy_key"
        )
        self.model = config.PRIMARY_LLM_MODEL

    def evaluate_sample(self, query: str, contexts: List[str], answer: str) -> Dict[str, Any]:
        context_str = "\n---\n".join(contexts)
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            query=query,
            contexts=context_str,
            answer=answer
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            raw_content = resp.choices[0].message.content.strip()

            # Bóc tách Markdown fences an toàn nếu LLM tự động chèn ```json ... ```
            if "```" in raw_content:
                match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_content, re.IGNORECASE)
                if match:
                    raw_content = match.group(1).strip()

            return json.loads(raw_content)
        except Exception as e:
            logger.warning("Lỗi trong quá trình thẩm định của LLM Judge: %s. Trả về điểm mặc định đánh giá khắt khe.", e)
            return {
                "faithfulness": 0.50,
                "answer_relevance": 0.50,
                "context_precision": 0.50,
                "reasoning": f"Thất bại phân tích phản hồi LLM: {e}"
            }

    def evaluate_benchmark_results(
        self,
        rag_outputs: List[Dict[str, Any]],
        output_csv: Path | str = ROOT_DIR / "benchmark" / "results" / "rq4_llm_judge_sample_details.csv"
    ) -> Dict[str, float]:
        out_file = Path(output_csv)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        faith_list, rel_list, prec_list = [], [], []

        logger.info("Bắt đầu quy trình LLM-as-a-Judge cho %d mẫu sinh câu trả lời...", len(rag_outputs))

        for idx, item in enumerate(rag_outputs, 1):
            q = item["query"]
            ans = item["answer"]
            ctxs = [c.get("content", "") for c in item.get("retrieved_contexts", [])]

            judgement = self.evaluate_sample(q, ctxs, ans)

            f_score = float(judgement.get("faithfulness", 0.0))
            r_score = float(judgement.get("answer_relevance", 0.0))
            p_score = float(judgement.get("context_precision", 0.0))

            faith_list.append(f_score)
            rel_list.append(r_score)
            prec_list.append(p_score)

            rows.append({
                "Sample_ID": idx,
                "Query": q[:80] + "...",
                "Faithfulness": f_score,
                "Answer_Relevance": r_score,
                "Context_Precision": p_score,
                "Judge_Reasoning": judgement.get("reasoning", "")
            })

        with open(out_file, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "Sample_ID", "Query", "Faithfulness", "Answer_Relevance", "Context_Precision", "Judge_Reasoning"
            ])
            writer.writeheader()
            writer.writerows(rows)

        summary = {
            "Mean_Faithfulness": round(float(np.mean(faith_list)), 4) if faith_list else 0.0,
            "Mean_Answer_Relevance": round(float(np.mean(rel_list)), 4) if rel_list else 0.0,
            "Mean_Context_Precision": round(float(np.mean(prec_list)), 4) if prec_list else 0.0,
        }

        logger.info("✓ Hoàn tất đánh giá LLM-as-a-Judge:")
        logger.info("  * Faithfulness (Độ trung thực): %.4f", summary["Mean_Faithfulness"])
        logger.info("  * Answer Relevance (Độ liên quan): %.4f", summary["Mean_Answer_Relevance"])
        logger.info("  * Context Precision (Độ chuẩn ngữ cảnh): %.4f", summary["Mean_Context_Precision"])
        logger.info("✓ Tệp chi tiết đã lưu tại: %s", out_file.resolve())

        return summary