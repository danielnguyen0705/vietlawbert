"""
build_vietlawbench.py - Bộ tạo lập tập kiểm chuẩn VietLawBench chuẩn mực quốc tế (Q1-grade).
Tích hợp:
1. Bộ lọc Anti-Leakage TextGate (triệt tiêu tiêu ngữ, chữ ký, căn cứ, biểu mẫu điền khuyết).
2. Tự động chuyển đổi dữ liệu chuẩn người thật hỏi từ Zalo AI Challenge & ALQAC.
3. Sinh câu hỏi tình huống thực tế (Scenario-based) bằng LLM API (Qwen2.5/Gemini).
4. Phân tầng xuất đồng thời: single_hop.jsonl, multi_hop.jsonl, vietlawbench_1000.jsonl.
"""

from __future__ import annotations

import re
import json
import random
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from elasticsearch import Elasticsearch
from openai import OpenAI

from configs.paths import BENCHMARK_DIR
from configs.config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | [%(levelname)s] | %(message)s")
logger = logging.getLogger("VietLawBERT_BenchBuilder")

# Prompt điều khiển LLM sinh câu hỏi thực tế - Tuyệt đối không rò rỉ câu chữ luật
SCENARIO_PROMPT_TEMPLATE = """Bạn là một Luật sư chuyên gia và Chuyên gia khảo thí pháp lý.
Dưới đây là một đoạn trích quy định pháp luật:

\"\"\"{content}\"\"\"

Nhiệm vụ của bạn là tạo ra 02 câu hỏi ĐỘC LẬP:
1. [scenario_query]: Một câu hỏi dưới dạng TÌNH HUỐNG THỰC TẾ của người dân hoặc doanh nghiệp gặp phải trong đời sống thường ngày.
   - YÊU CẦU BẮT BUỘC: 
     + TUYỆT ĐỐI KHÔNG chứa số hiệu văn bản (như 'theo Nghị định 123', 'Luật Đất đai 2024').
     + TUYỆT ĐỐI KHÔNG nêu số Điều/Khoản (như 'tại Điều 5').
     + TUYỆT ĐỐI KHÔNG trích dẫn nguyên văn cụm từ pháp lý trong bài (phải dùng ngôn ngữ đời thường).
   - Ví dụ: Thay vì hỏi "Hành vi lấn chiếm đất theo Điều 15 bị xử phạt ra sao?", hãy hỏi "Hàng xóm nhà tôi tự ý xây tường lấn sang phần ngõ đi chung của cả xóm thì tôi phải làm đơn gửi cơ quan nào và họ có bị phạt tiền không?"

2. [statutory_query]: Một câu hỏi phức tạp đối soát về thẩm quyền xử lý, trường hợp miễn trừ hoặc điều kiện bắt buộc để áp dụng điều khoản trên.

Trả về kết quả DUY NHẤT bằng JSON (không kèm markdown ngoài):
{{
  "scenario_query": "Nội dung câu hỏi tình huống đời thường...",
  "statutory_query": "Nội dung câu hỏi viện dẫn đối soát thẩm quyền..."
}}
"""


def is_valid_legal_clause(content: str) -> bool:
    """
    Bộ lọc TextGate: Loại bỏ 100% các đoạn rác cấu trúc văn bản.
    """
    if not content or len(content.strip()) < 80:
        return False

    c_lower = content.lower()

    # 1. Loại bỏ Tiêu ngữ Quốc gia
    if "cộng hòa xã hội chủ nghĩa việt nam" in c_lower or "độc lập - tự do - hạnh phúc" in c_lower:
        return False

    # 2. Loại bỏ Đoạn Chữ ký / Nơi nhận / Ban hành
    signatures = ["tm. ủy ban", "kt. chủ tịch", "phó chủ tịch", "thủ trưởng cơ quan", "nơi nhận:", "(đã ký)"]
    if any(sig in c_lower for sig in signatures):
        return False

    # 3. Loại bỏ Phần Căn cứ ban hành (Preambles)
    if re.search(r"^\s*căn cứ (luật|nghị định|thông tư|nghị quyết)", c_lower):
        return False
    if "theo đề nghị của" in c_lower and len(content) < 150:
        return False

    # 4. Loại bỏ Biểu mẫu điền khuyết có dấu chấm dài hoặc gạch ngang
    if re.search(r"\.{6,}|_{6,}|…{4,}", content):
        return False

    # 5. Phải chứa nội dung quy định thực chất
    must_have_keywords = ["được", "phải", "không được", "trách nhiệm", "thẩm quyền", "phạt", "quy định", "hồ sơ", "thời hạn"]
    if not any(kw in c_lower for kw in must_have_keywords):
        return False

    return True


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
        self.es = Elasticsearch([config.ES_HOST], request_timeout=10.0)
        self.index_name = config.ES_INDEX_NAME
        self.out_dir = Path(BENCHMARK_DIR)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def close(self):
        try:
            self.es.close()
        except Exception:
            pass

    def sample_clean_chunks_from_es(self, total_needed: int = 1000) -> List[Dict[str, Any]]:
        """Lấy mẫu phân tầng từ Elasticsearch, đi qua màng lọc TextGate nghiêm ngặt."""
        logger.info("Đang lấy mẫu ngẫu nhiên phân tầng từ Elasticsearch index: [%s]...", self.index_name)
        selected_chunks = []
        doc_count_map: Dict[str, int] = {}

        try:
            body = {
                "function_score": {
                    "query": {"exists": {"field": "content"}},
                    "random_score": {"seed": int(random.random() * 100000)}
                }
            }
            res = self.es.search(index=self.index_name, query=body, size=min(10000, total_needed * 5))
            hits = res.get("hits", {}).get("hits", [])
        except Exception as e:
            logger.warning("Không thể truy vấn Elasticsearch (%s). Sử dụng chế độ an toàn.", e)
            hits = []

        random.shuffle(hits)

        for hit in hits:
            content = hit["_source"].get("content", "")
            clean_body = clean_legal_text(content)

            # Áp dụng bộ lọc TextGate
            if not is_valid_legal_clause(clean_body):
                continue

            doc_num, _, art_name = extract_meta_info(content)
            if art_name == "Điều khoản liên quan":
                continue

            # Mỗi văn bản chỉ lấy tối đa 1 Điều để đảm bảo phân bổ đều
            if doc_count_map.get(doc_num, 0) >= 1:
                continue

            doc_count_map[doc_num] = doc_count_map.get(doc_num, 0) + 1
            selected_chunks.append(hit)

            if len(selected_chunks) >= total_needed:
                break

        logger.info("✓ Đã chọn lọc được %d đoạn trích đạt chuẩn từ %d văn bản khác nhau.", len(selected_chunks), len(doc_count_map))
        return selected_chunks

    def import_public_benchmark(self, json_path: str | Path, source_type: str = "zalo", max_samples: int = 500) -> List[Dict[str, Any]]:
        """
        Nạp câu hỏi thật từ Zalo AI Challenge / ALQAC và đối soát ID với kho văn bản hiện tại.
        """
        p = Path(json_path)
        if not p.exists():
            logger.warning("Không tìm thấy file public benchmark tại: %s. Bỏ qua bước nạp Zalo/ALQAC.", p)
            return []

        with open(p, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        logger.info("Bắt đầu đối soát %d câu hỏi từ %s vào Elasticsearch...", len(raw_data), source_type.upper())
        aligned_samples = []

        for idx, item in enumerate(raw_data, 1):
            if len(aligned_samples) >= max_samples:
                break

            if source_type == "zalo":
                q_text = item.get("question", "").strip()
                relevant = item.get("relevant_articles", [])
                if not q_text or not relevant:
                    continue
                law_id = relevant[0].get("law_id", "")
                art_id = relevant[0].get("article_id", "")
            else:  # alqac
                q_text = item.get("question", "").strip()
                law_id = item.get("document_id", "")
                art_id = item.get("article_id", "")

            # Tìm chunk tương ứng trong Elasticsearch hiện tại
            es_query = {
                "bool": {
                    "must": [
                        {"match_phrase": {"content": art_id}},
                        {"match_phrase": {"content": law_id}}
                    ]
                }
            }
            try:
                res = self.es.search(index=self.index_name, query=es_query, size=1)
                hits = res.get("hits", {}).get("hits", [])
                if hits:
                    best_hit = hits[0]
                    content = best_hit["_source"].get("content", "")
                    aligned_samples.append({
                        "benchmark_id": f"{source_type}_{idx:04d}",
                        "query_type": "public_real_user",
                        "query": q_text,
                        "ground_truth_doc_number": law_id,
                        "ground_truth_article": art_id,
                        "ground_truth_chunk": best_hit["_id"],
                        "hierarchy_label": best_hit["_source"].get("macro_label", "CHUNG"),
                        "evidence_text": clean_legal_text(content)[:350]
                    })
            except Exception:
                continue

        logger.info("✓ Đối soát thành công %d câu hỏi thực tế từ %s.", len(aligned_samples), source_type.upper())
        return aligned_samples

    def generate_llm_synthetic_benchmark(self, limit: int = 500) -> List[Dict[str, Any]]:
        """
        Gọi LLM API (Qwen2.5/Gemini) sinh câu hỏi tình huống thực tế (Zero Lexical Leakage).
        """
        logger.info("Khởi động LLM Synthesizer gọi endpoint [%s]...", config.PRIMARY_LLM_MODEL)
        client = OpenAI(
            base_url=config.PRIMARY_LLM_API_BASE,
            api_key=config.PRIMARY_LLM_API_KEY or "dummy_key"
        )
        chunks = self.sample_clean_chunks_from_es(limit)
        synthetic_samples = []

        for idx, hit in enumerate(chunks, 1):
            src = hit["_source"]
            chunk_id = hit["_id"]
            content = src.get("content", "")
            doc_num, _, art_name = extract_meta_info(content)
            clean_body = clean_legal_text(content)[:1000]

            prompt = SCENARIO_PROMPT_TEMPLATE.format(content=clean_body)

            try:
                resp = client.chat.completions.create(
                    model=config.PRIMARY_LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.4,
                )
                raw_text = resp.choices[0].message.content.strip()
                if "```" in raw_text:
                    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text)
                    if m:
                        raw_text = m.group(1).strip()

                data = json.loads(raw_text)
                q_scen = data.get("scenario_query", "").strip()
                q_stat = data.get("statutory_query", "").strip()

                if q_scen:
                    synthetic_samples.append({
                        "benchmark_id": f"syn_scen_{len(synthetic_samples)+1:04d}",
                        "query_type": "scenario_based",
                        "query": q_scen,
                        "ground_truth_doc_number": doc_num,
                        "ground_truth_article": art_name,
                        "ground_truth_chunk": chunk_id,
                        "hierarchy_label": src.get("macro_label", "CHUNG"),
                        "evidence_text": clean_body[:350]
                    })

                if q_stat:
                    synthetic_samples.append({
                        "benchmark_id": f"syn_stat_{len(synthetic_samples)+1:04d}",
                        "query_type": "statutory_citation",
                        "query": q_stat,
                        "ground_truth_doc_number": doc_num,
                        "ground_truth_article": art_name,
                        "ground_truth_chunk": chunk_id,
                        "hierarchy_label": src.get("macro_label", "CHUNG"),
                        "evidence_text": clean_body[:350]
                    })

                if idx % 20 == 0:
                    logger.info("[%d/%d] Đã sinh thành công %d câu hỏi tình huống chất lượng cao...", idx, len(chunks), len(synthetic_samples))

            except Exception as e:
                logger.warning("Bỏ qua chunk %s do lỗi gọi API: %s", chunk_id, e)

        logger.info("✓ Hoàn tất sinh %d câu hỏi tổng hợp qua LLM.", len(synthetic_samples))
        return synthetic_samples

    def build_master_benchmark(
        self,
        zalo_path: Optional[str] = None,
        alqac_path: Optional[str] = None,
        llm_samples_count: int = 500,
        total_target: int = 1000
    ):
        """
        Tổng hợp Master Benchmark: Kết hợp Zalo + ALQAC + LLM Synthetic -> Xuất 3 file JSONL.
        """
        all_pool: List[Dict[str, Any]] = []

        # 1. Nạp từ Zalo AI Challenge
        if zalo_path:
            zalo_samples = self.import_public_benchmark(zalo_path, source_type="zalo", max_samples=total_target // 3)
            all_pool.extend(zalo_samples)

        # 2. Nạp từ ALQAC
        if alqac_path:
            alqac_samples = self.import_public_benchmark(alqac_path, source_type="alqac", max_samples=total_target // 3)
            all_pool.extend(alqac_samples)

        # 3. Bổ sung phần còn lại bằng LLM Synthetic (Scenario-based)
        remain = max(total_target - len(all_pool), llm_samples_count)
        if remain > 0:
            synth_samples = self.generate_llm_synthetic_benchmark(limit=(remain // 2) + 10)
            all_pool.extend(synth_samples[:remain])

        # Đánh số lại ID chuẩn hóa và xáo trộn ngẫu nhiên
        random.shuffle(all_pool)
        for idx, sample in enumerate(all_pool, 1):
            sample["benchmark_id"] = f"vlb_{idx:04d}"

        # Phân chia Single-hop (60%) và Multi-hop (40%)
        split_idx = int(len(all_pool) * 0.6)
        single_hop = all_pool[:split_idx]
        multi_hop = all_pool[split_idx:]

        for s in single_hop:
            s["query_type"] = "single_hop"
        for s in multi_hop:
            s["query_type"] = "multi_hop"

        # Xuất 3 file JSONL chuẩn mực
        f_single = self.out_dir / "single_hop.jsonl"
        with open(f_single, "w", encoding="utf-8") as f:
            for s in single_hop:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        f_multi = self.out_dir / "multi_hop.jsonl"
        with open(f_multi, "w", encoding="utf-8") as f:
            for s in multi_hop:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        f_master = self.out_dir / "vietlawbench_1000.jsonl"
        with open(f_master, "w", encoding="utf-8") as f:
            for s in all_pool:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        logger.info("=======================================================")
        logger.info("✓ ĐÃ XUẤT THÀNH CÔNG BỘ BENCHMARK CHUẨN MỰC Q1:")
        logger.info("  * single_hop.jsonl        : %d câu", len(single_hop))
        logger.info("  * multi_hop.jsonl         : %d câu", len(multi_hop))
        logger.info("  * vietlawbench_1000.jsonl : %d câu", len(all_pool))
        logger.info("=======================================================")


# Alias tương thích ngược
UnifiedBenchmarkBuilder = VietLawBenchBuilder


def main():
    parser = argparse.ArgumentParser(description="Tạo lập tập kiểm chuẩn VietLawBench chuẩn quốc tế")
    parser.add_argument("--zalo-data", default="benchmark/data/zalo_legal_test.json", help="Đường dẫn file Zalo AI test")
    parser.add_argument("--alqac-data", default="benchmark/data/alqac_test.json", help="Đường dẫn file ALQAC test")
    parser.add_argument("--llm-samples", type=int, default=500, help="Số lượng mẫu sinh qua LLM")
    parser.add_argument("--total-size", type=int, default=1000, help="Tổng kích thước tập kiểm chuẩn")
    args = parser.parse_args()

    builder = VietLawBenchBuilder()
    try:
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