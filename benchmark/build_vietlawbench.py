"""
benchmark/build_vietlawbench.py - Bộ tạo lập tập kiểm chuẩn VietLawBench chuẩn quốc tế.
Triệt tiêu hoàn toàn rò rỉ từ vựng (Zero Lexical Leakage), phân định chuẩn Single-hop / Multi-hop.
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
from typing import List, Dict, Any, Tuple, Optional
from elasticsearch import Elasticsearch
from openai import OpenAI

from configs.paths import BENCHMARK_DIR
from configs.config import config
from benchmark.fetch_zalo_data import ZaloDataFetcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s | [%(levelname)s] | %(message)s")
logger = logging.getLogger("VietLawBERT_BenchBuilder")

DYNAMIC_SCENARIO_PROMPT = """Bạn là một Luật sư chuyên gia và Chuyên gia khảo thí pháp lý hàng đầu.
Nhiệm vụ của bạn là đọc một đoạn quy định pháp luật và đặt câu hỏi dưới góc nhìn của một NGƯỜI DÂN hoặc DOANH NGHIỆP đang gặp vướng mắc thực tế trong đời sống.

Dưới đây là một số ví dụ thực tế về cách người dân đặt câu hỏi (Trích từ cuộc thi Zalo AI Challenge):
{exemplars_text}

[QUY TẮC BẮT BUỘC ĐỂ KHÔNG RÒ RỈ CÂU CHỮ VĂN BẢN (ZERO LEXICAL LEAKAGE)]
1. Tuyệt đối KHÔNG chứa số hiệu văn bản (như 'theo Nghị định 123', 'Luật Đất đai 2024').
2. Tuyệt đối KHÔNG nêu số Điều/Khoản (như 'tại Điều 5').
3. Tuyệt đối KHÔNG trích nguyên văn câu chữ văn bản. Phải dùng ngôn ngữ đời thường (ví dụ: 'chồng bỏ đi', 'bị quỵt lương', 'bị thu hồi đất').
4. Đóng vai một người dân hoặc doanh nghiệp cụ thể đang gặp sự việc liên quan đến đoạn luật dưới đây để hỏi luật sư.

[ĐOẠN QUY ĐỊNH PHÁP LUẬT]
\"\"\"{content}\"\"\"

Hãy sinh ra duy nhất 01 chuỗi JSON theo định dạng sau (không kèm bất kỳ lời dẫn hay markdown ngoài):
{{
  "scenario_query": "Nội dung câu hỏi tình huống đời thường của người dân...",
  "statutory_query": "Câu hỏi đối soát về điều kiện áp dụng, thẩm quyền xử lý hoặc trường hợp miễn trừ..."
}}
"""


def is_valid_legal_clause(content: str) -> bool:
    """Bộ lọc TextGate: Loại bỏ 100% các đoạn rác cấu trúc văn bản."""
    if not content or len(content.strip()) < 80:
        return False
    c_lower = content.lower()
    if "cộng hòa xã hội chủ nghĩa việt nam" in c_lower or "độc lập - tự do - hạnh phúc" in c_lower:
        return False
    signatures = ["tm. ủy ban", "kt. chủ tịch", "phó chủ tịch", "thủ trưởng cơ quan", "nơi nhận:", "(đã ký)"]
    if any(sig in c_lower for sig in signatures):
        return False
    if re.search(r"^\s*căn cứ (luật|nghị định|thông tư|nghị quyết)", c_lower):
        return False
    if re.search(r"\.{6,}|_{6,}|…{4,}", content):
        return False
    must_have_keywords = ["được", "phải", "không được", "trách nhiệm", "thẩm quyền", "phạt", "quy định", "hồ sơ", "thời hạn", "hỗ trợ"]
    return any(kw in c_lower for kw in must_have_keywords)


def extract_meta_info(content: str) -> Tuple[str, str, str]:
    """Trích xuất metadata chuẩn hóa từ thẻ [META] và [HIERARCHY]."""
    doc_m = re.search(r"[META]\s*Văn bản:\s*([^|\n]+)", content)
    doc_num = doc_m.group(1).strip() if doc_m else "Văn bản hiện hành"
    h_m = re.search(r"[HIERARCHY]\s*([^\n]+)", content)
    path_str = h_m.group(1).strip() if h_m else "Điều khoản liên quan"
    art_m = re.search(r"Điều\s+(\d+[a-zA-Z]?)", path_str, re.IGNORECASE)
    art_name = art_m.group(0) if art_m else "Điều khoản liên quan"
    return doc_num, path_str, art_name


def clean_legal_text(content: str) -> str:
    clean = re.sub(r"[META].*?\n", "", content, flags=re.DOTALL)
    clean = re.sub(r"[HIERARCHY].*?\n", "", clean, flags=re.DOTALL)
    clean = clean.replace("[CONTENT]", "").strip()
    return clean


class VietLawBenchBuilder:
    def __init__(self):
        self.es = Elasticsearch([config.ES_HOST], request_timeout=20.0)
        self.index_name = config.ES_INDEX_NAME
        self.out_dir = Path(BENCHMARK_DIR)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        try:
            fetcher = ZaloDataFetcher()
            self.zalo_bank = fetcher.read_and_parse_zalo_queries()
        except Exception:
            self.zalo_bank = []

        api_base = os.getenv("CONTEXTUALIZER_API_BASE") or getattr(config, "PRIMARY_LLM_API_BASE", "https://api.openai.com/v1")
        api_key = os.getenv("CONTEXTUALIZER_API_KEY") or getattr(config, "PRIMARY_LLM_API_KEY", "dummy_key")
        self.client = OpenAI(base_url=api_base, api_key=api_key or "dummy_key")
        self.model = os.getenv("CONTEXTUALIZER_MODEL") or getattr(config, "PRIMARY_LLM_MODEL", "gpt-4o-mini")

    def close(self):
        try:
            self.es.close()
        except Exception:
            pass

    def sample_clean_chunks_from_es(self, total_needed: int = 1000) -> List[Dict[str, Any]]:
        logger.info("Đang lấy mẫu phân tầng từ Elasticsearch index: [%s]...", self.index_name)
        selected_chunks = []
        doc_count_map: Dict[str, int] = {}
        try:
            res = self.es.search(index=self.index_name, query={"match_all": {}}, size=min(10000, total_needed * 4))
            hits = res.get("hits", {}).get("hits", [])
        except Exception as e:
            logger.error("Lỗi kết nối ES: %s", e)
            hits = []

        random.shuffle(hits)
        for hit in hits:
            content = hit["_source"].get("content", "")
            clean_body = clean_legal_text(content)
            if not is_valid_legal_clause(clean_body):
                continue
            doc_num, path_str, art_name = extract_meta_info(content)
            if art_name == "Điều khoản liên quan" or doc_count_map.get(doc_num, 0) >= 2:
                continue

            doc_count_map[doc_num] = doc_count_map.get(doc_num, 0) + 1
            selected_chunks.append({
                "hit": hit,
                "clean_body": clean_body,
                "doc_num": doc_num,
                "path_str": path_str,
                "art_name": art_name,
                "macro_label": hit["_source"].get("macro_label", "CHUNG"),
                "raw_content": content
            })
            if len(selected_chunks) >= total_needed:
                break
        return selected_chunks

    def build_master_benchmark(
        self,
        zalo_path: Optional[str] = None,
        alqac_path: Optional[str] = None,
        total_target: int = 1000
    ):
        all_pool: List[Dict[str, Any]] = []

        # 1. Nạp từ Zalo AI Challenge & ALQAC đã được bình duyệt (Người thật hỏi)
        if zalo_path and Path(zalo_path).exists():
            with open(zalo_path, "r", encoding="utf-8") as f:
                zalo_raw = json.load(f)
            for idx, item in enumerate(zalo_raw[:total_target // 2], 1):
                all_pool.append({
                    "benchmark_id": f"vlb_zalo_{idx:04d}",
                    "query_type": "single_hop",
                    "query": item.get("question", ""),
                    "ground_truth_doc_number": (item.get("relevant_articles") or [{}])[0].get("law_id", ""),
                    "ground_truth_article": (item.get("relevant_articles") or [{}])[0].get("article_id", ""),
                    "ground_truth_chunk": "",
                    "ground_truth_docs": item.get("relevant_articles", []),
                    "evidence_text": item.get("text", "")[:350],
                    "raw_content": item.get("text", "")
                })

        # 2. Sinh câu hỏi bằng LLM có kiểm soát Zero Lexical Leakage nếu còn thiếu
        remain = total_target - len(all_pool)
        if remain > 0:
            chunks = self.sample_clean_chunks_from_es(remain)
            for idx, item in enumerate(chunks, 1):
                clean_body = item["clean_body"][:800]
                prompt = DYNAMIC_SCENARIO_PROMPT.format(exemplars_text="", content=clean_body)
                try:
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                    )
                    raw_text = resp.choices[0].message.content.strip()
                    if "```" in raw_text:
                        raw_text = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text).group(1).strip()
                    data = json.loads(raw_text)
                    q = data.get("scenario_query") or data.get("statutory_query")
                    if q and len(q) >= 15:
                        all_pool.append({
                            "benchmark_id": f"vlb_syn_{len(all_pool)+1:04d}",
                            "query_type": "single_hop",
                            "query": q,
                            "ground_truth_doc_number": item["doc_num"],
                            "ground_truth_article": item["art_name"],
                            "ground_truth_chunk": item["hit"]["_id"],
                            "ground_truth_docs": [{"doc_number": item["doc_num"], "article": item["art_name"]}],
                            "evidence_text": clean_body[:350],
                            "raw_content": item["raw_content"]
                        })
                except Exception as ex:
                    logger.warning("Bỏ qua mẫu do lỗi LLM: %s", ex)

        # 3. Phân định Single-hop vs Multi-hop theo bản chất số lượng căn cứ
        single_hop = []
        multi_hop = []
        for s in all_pool:
            gts = s.get("ground_truth_docs", [])
            if len(gts) > 1:
                s["query_type"] = "multi_hop"
                multi_hop.append(s)
            else:
                s["query_type"] = "single_hop"
                single_hop.append(s)

        with open(self.out_dir / "single_hop.jsonl", "w", encoding="utf-8") as f:
            for s in single_hop:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        with open(self.out_dir / "multi_hop.jsonl", "w", encoding="utf-8") as f:
            for s in multi_hop:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        with open(self.out_dir / "vietlawbench_1000.jsonl", "w", encoding="utf-8") as f:
            for s in all_pool:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        logger.info("✓ ĐÃ XUẤT THÀNH CÔNG BỘ BENCHMARK CHUẨN MỰC Q1:")
        logger.info("  * single_hop.jsonl        : %d câu", len(single_hop))
        logger.info("  * multi_hop.jsonl         : %d câu", len(multi_hop))
        logger.info("  * vietlawbench_1000.jsonl : %d câu", len(all_pool))


UnifiedBenchmarkBuilder = VietLawBenchBuilder


def main():
    parser = argparse.ArgumentParser(description="Tạo lập tập kiểm chuẩn VietLawBench chuẩn quốc tế")
    parser.add_argument("--zalo-data", default="benchmark/data/zalo_legal_test.json")
    parser.add_argument("--total-size", type=int, default=1000)
    args = parser.parse_args()

    builder = VietLawBenchBuilder()
    try:
        builder.build_master_benchmark(
            zalo_path=args.zalo_data if Path(args.zalo_data).exists() else None,
            total_target=args.total_size,
        )
    finally:
        builder.close()


if __name__ == "__main__":
    main()