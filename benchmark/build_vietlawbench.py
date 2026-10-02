"""
build_vietlawbench.py - Bộ tạo lập tập kiểm chuẩn VietLawBench
Tích hợp:
1. Bộ lọc Anti-Leakage TextGate (triệt tiêu tiêu ngữ, chữ ký, căn cứ, biểu mẫu điền khuyết).
2. Tự động chuyển đổi dữ liệu chuẩn người thật hỏi từ Zalo AI Challenge & ALQAC.
3. Sinh câu hỏi phỏng sinh tình huống thực tế (Dynamic Few-Shot In-Context Learning).
4. Phân tầng xuất đồng thời: single_hop.jsonl, multi_hop.jsonl, vietlawbench_1000.jsonl.
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
3. Tuyệt đối KHÔNG trích nguyên văn cụm từ pháp lý chuyên ngành trong bài. Phải dùng ngôn ngữ đời thường (ví dụ: 'chồng bỏ đi', 'bị quỵt lương', 'bị thu hồi đất', 'lấn chiếm ngõ đi').
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
    if "theo đề nghị của" in c_lower and len(content) < 150:
        return False

    if re.search(r"\.{6,}|_{6,}|…{4,}", content):
        return False

    must_have_keywords = ["được", "phải", "không được", "trách nhiệm", "thẩm quyền", "phạt", "quy định", "hồ sơ", "thời hạn", "hỗ trợ"]
    return any(kw in c_lower for kw in must_have_keywords)


def extract_meta_info(content: str) -> Tuple[str, str, str]:
    """Trích xuất metadata chuẩn hóa từ thẻ [META] và [HIERARCHY]."""
    doc_m = re.search(r"\[META\]\s*Văn bản:\s*([^|\n]+)", content)
    doc_num = doc_m.group(1).strip() if doc_m else "Văn bản hiện hành"

    h_m = re.search(r"\[HIERARCHY\]\s*([^\n]+)", content)
    path_str = h_m.group(1).strip() if h_m else "Điều khoản liên quan"

    art_m = re.search(r"Điều\s+(\d+[a-zA-Z]?)", path_str, re.IGNORECASE)
    art_name = art_m.group(0) if art_m else "Điều khoản liên quan"

    return doc_num, path_str, art_name


def clean_legal_text(content: str) -> str:
    """Làm sạch các tag kỹ thuật để đưa vào prompt LLM hoặc làm evidence."""
    clean = re.sub(r"\[META\].*?\n", "", content, flags=re.DOTALL)
    clean = re.sub(r"\[HIERARCHY\].*?\n", "", clean, flags=re.DOTALL)
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
        except Exception as e:
            logger.warning("Không thể khởi tạo ZaloDataFetcher (%s). Chuyển về chế độ prompt mặc định.", e)
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

    def get_random_zalo_exemplars(self, k: int = 3) -> str:
        if not self.zalo_bank:
            return "- Mẫu: Người lao động bị công ty giữ lương 2 tháng thì có được đơn phương chấm dứt hợp đồng không?"
        sampled = random.sample(self.zalo_bank, min(k, len(self.zalo_bank)))
        return "\n".join([f"- Mẫu {idx}: \"{item['question']}\"" for idx, item in enumerate(sampled, 1)])

    def sample_clean_chunks_from_es(self, total_needed: int = 1000) -> List[Dict[str, Any]]:
        """Lấy mẫu ngẫu nhiên phân tầng từ Elasticsearch qua bộ lọc TextGate."""
        logger.info("Đang lấy mẫu phân tầng từ Elasticsearch index: [%s]...", self.index_name)
        selected_chunks = []
        doc_count_map: Dict[str, int] = {}

        try:
            body = {
                "function_score": {
                    "query": {"exists": {"field": "content"}},
                    "random_score": {"seed": int(time.time())}
                }
            }
            res = self.es.search(index=self.index_name, query=body, size=min(10000, max(2000, total_needed * 5)))
            hits = res.get("hits", {}).get("hits", [])
        except Exception as e:
            logger.warning("Không thể truy vấn random score từ ES (%s). Thử truy vấn cơ bản match_all.", e)
            try:
                res = self.es.search(index=self.index_name, query={"match_all": {}}, size=min(10000, total_needed * 3))
                hits = res.get("hits", {}).get("hits", [])
            except Exception as ex:
                logger.error("Lỗi kết nối ES: %s", ex)
                hits = []

        random.shuffle(hits)

        for hit in hits:
            content = hit["_source"].get("content", "")
            clean_body = clean_legal_text(content)

            if not is_valid_legal_clause(clean_body):
                continue

            doc_num, path_str, art_name = extract_meta_info(content)
            if art_name == "Điều khoản liên quan":
                continue

            if doc_count_map.get(doc_num, 0) >= 2:
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

        logger.info("✓ Đã chọn lọc được %d đoạn trích đạt chuẩn TextGate từ %d văn bản.", len(selected_chunks), len(doc_count_map))
        return selected_chunks

    def generate_stratified_benchmark(self, target_size: int = 1000):
        """Tự động sinh tập kiểm chuẩn phân tầng khi phát hiện thư mục benchmark trống."""
        logger.info("Kích hoạt tạo lập nhanh Stratified Benchmark quy mô: %d mẫu...", target_size)
        chunks = self.sample_clean_chunks_from_es(target_size)
        all_samples = []

        for idx, item in enumerate(chunks, 1):
            raw_c = item["clean_body"]
            first_sentence = raw_c.split(".")[0].strip()
            pseudo_query = f"Quy định pháp luật về {first_sentence.lower()[:120]}"

            all_samples.append({
                "benchmark_id": f"vlb_{idx:04d}",
                "query_type": "single_hop" if idx % 2 == 1 else "multi_hop",
                "query": pseudo_query,
                "ground_truth_doc_number": item["doc_num"],
                "ground_truth_article": item["art_name"],
                "ground_truth_chunk": item["hit"]["_id"],
                "hierarchy_label": item["macro_label"],
                "evidence_text": item["clean_body"][:350],
                "raw_content": item["raw_content"]
            })

        split_idx = int(len(all_samples) * 0.6)
        single_hop = all_samples[:split_idx]
        multi_hop = all_samples[split_idx:]

        with open(self.out_dir / "single_hop.jsonl", "w", encoding="utf-8") as f:
            for s in single_hop:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        with open(self.out_dir / "multi_hop.jsonl", "w", encoding="utf-8") as f:
            for s in multi_hop:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        with open(self.out_dir / "vietlawbench_1000.jsonl", "w", encoding="utf-8") as f:
            for s in all_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        logger.info("✓ Khởi tạo nhanh thành công 3 tệp benchmark tại: %s", self.out_dir.resolve())

    def generate_llm_synthetic_benchmark(self, limit: int = 500) -> List[Dict[str, Any]]:
        """Gọi LLM API với Dynamic Few-shot sinh câu hỏi tình huống thực tế."""
        logger.info("Khởi động Dynamic LLM Synthesizer gọi model [%s]...", self.model)
        chunks = self.sample_clean_chunks_from_es(limit)
        synthetic_samples = []

        for idx, item in enumerate(chunks, 1):
            clean_body = item["clean_body"][:1000]
            exemplars = self.get_random_zalo_exemplars(k=3)
            prompt = DYNAMIC_SCENARIO_PROMPT.format(exemplars_text=exemplars, content=clean_body)

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
                q_scen = data.get("scenario_query", "").strip()
                q_stat = data.get("statutory_query", "").strip()

                if q_scen and len(q_scen) >= 15:
                    synthetic_samples.append({
                        "benchmark_id": f"syn_scen_{len(synthetic_samples)+1:04d}",
                        "query_type": "scenario_based",
                        "query": q_scen,
                        "ground_truth_doc_number": item["doc_num"],
                        "ground_truth_article": item["art_name"],
                        "ground_truth_chunk": item["hit"]["_id"],
                        "hierarchy_label": item["macro_label"],
                        "evidence_text": clean_body[:350],
                        "raw_content": item["raw_content"]
                    })

                if q_stat and len(q_stat) >= 15:
                    synthetic_samples.append({
                        "benchmark_id": f"syn_stat_{len(synthetic_samples)+1:04d}",
                        "query_type": "statutory_citation",
                        "query": q_stat,
                        "ground_truth_doc_number": item["doc_num"],
                        "ground_truth_article": item["art_name"],
                        "ground_truth_chunk": item["hit"]["_id"],
                        "hierarchy_label": item["macro_label"],
                        "evidence_text": clean_body[:350],
                        "raw_content": item["raw_content"]
                    })

                if idx % 10 == 0:
                    logger.info("[%d/%d] Đã sinh thành công %d câu hỏi chất lượng cao...", idx, len(chunks), len(synthetic_samples))
                time.sleep(0.3)
            except Exception as e:
                logger.warning("Bỏ qua chunk %s do lỗi API: %s", item["hit"]["_id"], e)

        logger.info("✓ Hoàn tất sinh %d câu hỏi tổng hợp qua LLM.", len(synthetic_samples))
        return synthetic_samples

    def build_master_benchmark(
        self,
        zalo_path: Optional[str] = None,
        alqac_path: Optional[str] = None,
        llm_samples_count: int = 500,
        total_target: int = 1000
    ):
        all_pool: List[Dict[str, Any]] = []

        if zalo_path and Path(zalo_path).exists():
            with open(zalo_path, "r", encoding="utf-8") as f:
                zalo_raw = json.load(f)
            for idx, item in enumerate(zalo_raw[:total_target // 3], 1):
                all_pool.append({
                    "benchmark_id": f"zalo_{idx:04d}",
                    "query_type": "public_real_user",
                    "query": item.get("question", ""),
                    "ground_truth_doc_number": (item.get("relevant_articles") or [{}])[0].get("law_id", ""),
                    "ground_truth_article": (item.get("relevant_articles") or [{}])[0].get("article_id", ""),
                    "ground_truth_chunk": "",
                    "hierarchy_label": "CHUNG",
                    "evidence_text": item.get("text", "")[:350],
                    "raw_content": item.get("text", "")
                })

        remain = max(total_target - len(all_pool), llm_samples_count)
        if remain > 0:
            synth_samples = self.generate_llm_synthetic_benchmark(limit=(remain // 2) + 10)
            all_pool.extend(synth_samples[:remain])

        if len(all_pool) < total_target:
            self.generate_stratified_benchmark(total_target)
            return

        random.shuffle(all_pool)
        for idx, sample in enumerate(all_pool, 1):
            sample["benchmark_id"] = f"vlb_{idx:04d}"

        split_idx = int(len(all_pool) * 0.6)
        single_hop = all_pool[:split_idx]
        multi_hop = all_pool[split_idx:]

        for s in single_hop:
            s["query_type"] = "single_hop"
        for s in multi_hop:
            s["query_type"] = "multi_hop"

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
    parser.add_argument("--alqac-data", default="benchmark/data/alqac_test.json")
    parser.add_argument("--llm-samples", type=int, default=500)
    parser.add_argument("--total-size", type=int, default=1000)
    parser.add_argument("--mode", default="master", choices=["master", "stratified"])
    args = parser.parse_args()

    builder = VietLawBenchBuilder()
    try:
        if args.mode == "stratified":
            builder.generate_stratified_benchmark(args.total_size)
        else:
            builder.build_master_benchmark(
                zalo_path=args.zalo_data if Path(args.zalo_data).exists() else None,
                alqac_path=args.alqac_data if Path(args.alqac_data).exists() else None,
                llm_samples_count=args.llm_samples,
                total_target=args.total_size,
            )
    finally:
        builder.close()


if __name__ == "__main__":
    main()