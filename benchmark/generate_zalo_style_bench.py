"""
generate_zalo_style_bench.py - Động cơ sinh câu hỏi pháp lý phỏng sinh phong cách Zalo AI Challenge.
Cơ chế: Dynamic Few-Shot In-Context Learning.
- Quét sâu Elasticsearch đảm bảo gom ĐỦ số lượng Điều luật từ các văn bản khác nhau.
- Bơm 3 mẫu Zalo thật vào prompt để LLM hóa thân thành người dân hỏi luật sư.
- Tích hợp kiểm soát tốc độ gọi API an toàn.
"""

from __future__ import annotations

import os
import re
import json
import time
import random
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from elasticsearch import Elasticsearch
from openai import OpenAI

from configs.paths import BENCHMARK_DIR
from configs.config import config
from benchmark.fetch_zalo_data import ZaloDataFetcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s | [%(levelname)s] | %(message)s")
logger = logging.getLogger("ZaloStyleGenerator")


DYNAMIC_PROMPT_TEMPLATE = """Bạn là một Chuyên gia ngôn ngữ học và Khảo thí pháp luật.
Nhiệm vụ của bạn là đọc một đoạn quy định pháp luật và đặt câu hỏi dưới góc nhìn của một NGƯỜI DÂN đang gặp vướng mắc thực tế.

Dưới đây là một số ví dụ thực tế về cách người dân đặt câu hỏi (Trích từ cuộc thi Zalo AI Challenge):
{exemplars_text}

[QUY TẮC BẮT BUỘC ĐỂ GIỐNG ZALO AI CHALLENGE]
1. Tuyệt đối KHÔNG nhắc đến số hiệu văn bản (như "theo Nghị định 123", "Luật Đất đai").
2. Tuyệt đối KHÔNG nêu số Điều/Khoản (như "Điều 5 quy định gì?").
3. Tuyệt đối KHÔNG trích nguyên văn câu chữ trong văn bản luật. Phải dùng ngôn từ đời sống (ví dụ: "chồng bỏ đi", "bị quỵt lương", "bị giữ xe", "lấn chiếm ngõ").
4. Đóng vai một người dân hoặc doanh nghiệp cụ thể đang gặp sự việc liên quan đến đoạn luật dưới đây để hỏi luật sư.

[ĐOẠN LUẬT QUY ĐỊNH]
\"\"\"{content}\"\"\"

Hãy sinh ra duy nhất 01 chuỗi JSON theo định dạng sau (không kèm markdown ngoài):
{{"scenario_query": "Nội dung câu hỏi tình huống đời thường của người dân..."}}
"""


def is_valid_clause(content: str) -> bool:
    """Loại bỏ tiêu ngữ, căn cứ ban hành, chữ ký và văn bản rác."""
    if not content or len(content.strip()) < 80:
        return False
    c_low = content.lower()
    if "cộng hòa xã hội chủ nghĩa việt nam" in c_low or "độc lập - tự do" in c_low:
        return False
    if any(sig in c_low for sig in ["tm. ủy ban", "kt. chủ tịch", "phó chủ tịch", "nơi nhận:", "(đã ký)"]):
        return False
    if re.search(r"^\s*căn cứ (luật|nghị định|thông tư|nghị quyết)", c_low):
        return False
    if re.search(r"\.{6,}|_{6,}|…{4,}", content):
        return False
    return any(kw in c_low for kw in ["được", "phải", "không được", "trách nhiệm", "thẩm quyền", "phạt", "thời hạn", "hỗ trợ"])


def extract_meta(content: str) -> Tuple[str, str]:
    doc_m = re.search(r"\[META\]\s*Văn bản:\s*([^|\n]+)", content)
    doc_num = doc_m.group(1).strip() if doc_m else "Văn bản hiện hành"
    h_m = re.search(r"\[HIERARCHY\]\s*([^\n]+)", content)
    path_str = h_m.group(1).strip() if h_m else "Điều khoản liên quan"
    art_m = re.search(r"Điều\s+(\d+[a-zA-Z]?)", path_str, re.IGNORECASE)
    art_name = art_m.group(0) if art_m else "Điều khoản liên quan"
    return doc_num, art_name


def clean_content(content: str) -> str:
    clean = re.sub(r"\[META\].*?\n", "", content, flags=re.DOTALL)
    clean = re.sub(r"\[HIERARCHY\].*?\n", "", clean, flags=re.DOTALL)
    return clean.replace("[CONTENT]", "").strip()


class ZaloStyleBenchGenerator:
    def __init__(self):
        self.es = Elasticsearch([config.ES_HOST], request_timeout=30.0)
        self.index_name = config.ES_INDEX_NAME
        self.out_dir = Path(BENCHMARK_DIR)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        fetcher = ZaloDataFetcher()
        self.zalo_bank = fetcher.read_and_parse_zalo_queries()
        logger.info("✓ Ngân hàng mẫu Zalo AI sẵn sàng với %d câu hỏi phong cách thật.", len(self.zalo_bank))

        api_key = config.PRIMARY_LLM_API_KEY or os.getenv("PRIMARY_LLM_API_KEY") or "dummy_key"
        self.client = OpenAI(
            base_url=config.PRIMARY_LLM_API_BASE,
            api_key=api_key
        )
        self.model = config.PRIMARY_LLM_MODEL

    def get_random_zalo_exemplars(self, k: int = 3) -> str:
        sampled = random.sample(self.zalo_bank, min(k, len(self.zalo_bank)))
        return "\n".join([f"- Mẫu {idx}: \"{item['question']}\"" for idx, item in enumerate(sampled, 1)])

    def fetch_candidate_law_chunks(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Quét sâu và phân tán qua Elasticsearch để bảo đảm gom ĐỦ số lượng Điều luật khác nhau."""
        logger.info("Đang quét Elasticsearch để gom đủ %d Điều luật đa dạng...", limit)
        valid_chunks = []
        doc_tracker: Dict[str, int] = {}
        
        # Quét theo batch lớn 2000 hits để không bị co cụm vào vài văn bản
        query_dsl = {
            "function_score": {
                "query": {"exists": {"field": "content"}},
                "random_score": {"seed": random.randint(1, 1000000)}
            }
        }

        try:
            res = self.es.search(
                index=self.index_name,
                query=query_dsl,
                size=min(10000, max(2000, limit * 15))
            )
            hits = res.get("hits", {}).get("hits", [])
        except Exception:
            res = self.es.search(
                index=self.index_name,
                query={"match_all": {}},
                size=min(10000, max(2000, limit * 15))
            )
            hits = res.get("hits", {}).get("hits", [])

        random.shuffle(hits)

        for h in hits:
            raw_c = h["_source"].get("content", "")
            clean_c = clean_content(raw_c)
            if not is_valid_clause(clean_c):
                continue

            doc_num, art = extract_meta(raw_c)
            # Khống chế mỗi văn bản chỉ lấy tối đa 1-2 Điều
            max_per_doc = 1 if limit <= 100 else 2
            if art == "Điều khoản liên quan" or doc_tracker.get(doc_num, 0) >= max_per_doc:
                continue

            doc_tracker[doc_num] = doc_tracker.get(doc_num, 0) + 1
            valid_chunks.append({
                "chunk_id": h["_id"],
                "content": clean_c,
                "doc_number": doc_num,
                "article": art,
                "macro_label": h["_source"].get("macro_label", "CHUNG")
            })

            if len(valid_chunks) >= limit:
                break

        logger.info("✓ Đã gom đủ %d Điều luật hợp lệ từ %d văn bản hoàn toàn khác nhau.", len(valid_chunks), len(doc_tracker))
        return valid_chunks

    def generate_benchmark(self, num_questions: int = 50, output_filename: str = "test_50.jsonl"):
        chunks = self.fetch_candidate_law_chunks(limit=num_questions)
        generated_records = []
        out_file = self.out_dir / output_filename

        logger.info("Bắt đầu sinh tuần tự %d câu hỏi phong cách Zalo AI...", len(chunks))

        for idx, chunk in enumerate(chunks, 1):
            exemplars = self.get_random_zalo_exemplars(k=3)
            prompt = DYNAMIC_PROMPT_TEMPLATE.format(
                exemplars_text=exemplars,
                content=chunk["content"][:1000]
            )

            success = False
            for retry in range(3):
                try:
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                    )
                    raw_text = resp.choices[0].message.content.strip()
                    if "```" in raw_text:
                        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text)
                        if m:
                            raw_text = m.group(1).strip()

                    data = json.loads(raw_text)
                    query_text = data.get("scenario_query", "").strip()

                    if query_text and len(query_text) >= 15:
                        record = {
                            "benchmark_id": f"ques_{len(generated_records)+1:04d}",
                            "query_type": "scenario_based",
                            "query": query_text,
                            "ground_truth_doc_number": chunk["doc_number"],
                            "ground_truth_article": chunk["article"],
                            "ground_truth_chunk": chunk["chunk_id"],
                            "hierarchy_label": chunk["macro_label"],
                            "evidence_text": chunk["content"][:300]
                        }
                        generated_records.append(record)
                        success = True
                        break
                except Exception as e:
                    logger.warning("Thử lại lần %d cho chunk %s do: %s", retry + 1, chunk["chunk_id"], e)
                    time.sleep(2.0)

            if not success:
                logger.error("Bỏ qua chunk %s sau 3 lần thử.", chunk["chunk_id"])

            if idx % 5 == 0 or idx == len(chunks):
                logger.info("Tiến độ: [%d/%d] câu hỏi đã sinh thành công.", len(generated_records), len(chunks))

            # Nghỉ nhẹ 0.5s để bảo vệ hạn ngạch API (Rate limit)
            time.sleep(0.5)

        with open(out_file, "w", encoding="utf-8") as f:
            for item in generated_records:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info("=======================================================")
        logger.info("✓ HOÀN TẤT: Đã xuất trọn vẹn %d/%d câu hỏi tại: %s", len(generated_records), num_questions, out_file.resolve())
        logger.info("=======================================================")


def main():
    parser = argparse.ArgumentParser(description="Sinh dữ liệu kiểm chuẩn phỏng sinh Zalo AI Challenge")
    parser.add_argument("--num-queries", type=int, default=50, help="Số lượng câu hỏi cần sinh")
    parser.add_argument("--output", type=str, default="test_50.jsonl")
    args = parser.parse_args()

    generator = ZaloStyleBenchGenerator()
    generator.generate_benchmark(num_questions=args.num_queries, output_filename=args.output)


if __name__ == "__main__":
    main()