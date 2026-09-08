"""
build_vietlawbench.py - Bộ sinh tập dữ liệu đánh giá thực nghiệm chuẩn hóa VietLawBench (1.000 mẫu).
Trích xuất tự động từ Topo đồ thị Neo4j: 600 câu Single-hop + 400 câu Multi-hop Legal Reasoning.
"""

from __future__ import annotations

import re
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any
from neo4j import GraphDatabase

from configs.paths import BENCHMARK_DIR
from configs.config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | [%(levelname)s] | %(message)s")
logger = logging.getLogger("VietLawBERT_BenchBuilder")


def extract_key_phrase(text: str) -> str:
    """Rút trích mệnh đề pháp lý cốt lõi từ nội dung quy định."""
    clean = re.sub(r"\[META\].*?\n", "", text, flags=re.DOTALL)
    clean = re.sub(r"\[HIERARCHY\].*?\n", "", clean, flags=re.DOTALL)
    clean = re.sub(r"\[CONTENT\]", "", clean).strip()
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    first_line = lines[0] if lines else clean[:120]
    first_line = re.sub(r"^(\d+|[a-zđ])[\.\)\s\-]+", "", first_line, flags=re.IGNORECASE).strip()
    return first_line[:120]


class VietLawBenchBuilder:
    def __init__(self, uri: str = None, user: str = None, password: str = None):
        self.uri = uri or config.NEO4J_URI
        self.user = user or config.NEO4J_USER
        self.password = password or config.NEO4J_PASSWORD
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self.benchmark_dir = Path(BENCHMARK_DIR)
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)

    def close(self):
        self.driver.close()

    def generate_single_hop_samples(self, limit: int = 600) -> List[Dict[str, Any]]:
        """Sinh các truy vấn đơn tầng (Single-hop) dựa trên nội dung trực tiếp của Điều/Khoản."""
        cypher = """
        MATCH (d:LawDocument)-[:HAS_CHUNK]->(c:Chunk)
        WHERE c.content IS NOT NULL AND size(c.content) > 100
        RETURN d.doc_number AS doc_number, d.title AS doc_title,
               c.chunk_id AS ground_truth_chunk, c.hierarchy_path AS path,
               c.macro_label AS hierarchy_label, c.content AS content
        LIMIT $limit
        """
        samples = []
        templates = [
            "Theo quy định tại {path} thuộc {doc_number}, {topic} được quy định như thế nào?",
            "Căn cứ vào {path} của {doc_number}, nội dung điều chỉnh về {topic} là gì?",
            "Quy định pháp lý về {topic} được nêu tại {path} ({doc_number}) ra sao?",
        ]
        with self.driver.session() as session:
            records = session.run(cypher, limit=limit * 2).data()

        for idx, record in enumerate(records):
            doc_num = record["doc_number"] or "văn bản hiện hành"
            path_str = record["path"] or "Quy định pháp lý"
            content = record["content"] or ""
            topic = extract_key_phrase(content)
            if len(topic) < 15:
                continue

            tpl = templates[idx % len(templates)]
            query = tpl.format(path=path_str, doc_number=doc_num, topic=topic)

            art_match = re.search(r"Điều\s+(\d+[a-zA-Z]?)", path_str)
            art_name = art_match.group(0) if art_match else "Điều khoản liên quan"

            samples.append({
                "benchmark_id": f"single_hop_{len(samples)+1:04d}",
                "query_type": "single_hop",
                "query": query,
                "ground_truth_doc_number": doc_num,
                "ground_truth_article": art_name,
                "ground_truth_chunk": record["ground_truth_chunk"],
                "hierarchy_label": record["hierarchy_label"],
                "evidence_text": content[:300]
            })
            if len(samples) >= limit:
                break

        logger.info(f"Đã sinh thành công {len(samples)} mẫu kiểm tra Single-hop.")
        return samples

    def generate_multi_hop_samples(self, limit: int = 400) -> List[Dict[str, Any]]:
        """Sinh các truy vấn đa tầng (Multi-hop) dựa trên các cạnh quan hệ liên văn bản."""
        cypher = """
        MATCH (d1:LawDocument)-[r:LEGAL_RELATION]->(d2:LawDocument)-[:HAS_CHUNK]->(c2:Chunk)
        WHERE size(c2.content) > 100 AND d1.doc_number <> d2.doc_number
        RETURN d1.doc_number AS doc_a, r.type AS rel_type, d2.doc_number AS doc_b,
               c2.chunk_id AS ground_truth_chunk, c2.hierarchy_path AS path,
               c2.macro_label AS hierarchy_label, c2.content AS content
        LIMIT $limit
        """
        samples = []
        with self.driver.session() as session:
            records = session.run(cypher, limit=limit * 2).data()

        for idx, record in enumerate(records):
            rel = record["rel_type"] or "liên quan đến"
            doc_a = record["doc_a"] or "Văn bản A"
            doc_b = record["doc_b"] or "Văn bản B"
            path_str = record["path"] or "Điều khoản thi hành"

            art_match = re.search(r"Điều\s+(\d+[a-zA-Z]?)", path_str)
            art_name = art_match.group(0) if art_match else "Điều khoản liên quan"

            query = (
                f"Căn cứ theo mối quan hệ {rel} giữa {doc_a} và {doc_b}, "
                f"nội dung quy định phối hợp tại {path_str} được giải quyết như thế nào?"
            )
            samples.append({
                "benchmark_id": f"multi_hop_{len(samples)+1:04d}",
                "query_type": "multi_hop",
                "query": query,
                "ground_truth_doc_number": doc_b,
                "ground_truth_article": art_name,
                "ground_truth_docs": [
                    {"doc_number": doc_a, "article": "Căn cứ dẫn chiếu"},
                    {"doc_number": doc_b, "article": art_name}
                ],
                "ground_truth_chunk": record["ground_truth_chunk"],
                "hierarchy_label": record["hierarchy_label"],
                "reasoning_chain": [doc_a, rel, doc_b],
                "evidence_text": record["content"][:300]
            })
            if len(samples) >= limit:
                break

        logger.info(f"Đã sinh thành công {len(samples)} mẫu kiểm tra Multi-hop.")
        return samples

    def build_and_export_all(self, single_limit: int = 600, multi_limit: int = 400):
        single_samples = self.generate_single_hop_samples(single_limit)
        multi_samples = self.generate_multi_hop_samples(multi_limit)
        all_samples = single_samples + multi_samples

        def save_jsonl(path: Path, data: List[Dict[str, Any]]):
            with open(path, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        save_jsonl(self.benchmark_dir / "vietlawbench_1000.jsonl", all_samples)
        save_jsonl(self.benchmark_dir / "single_hop.jsonl", single_samples)
        save_jsonl(self.benchmark_dir / "multi_hop.jsonl", multi_samples)

        logger.info(f"Đã lưu trọn bộ Benchmark tại: {self.benchmark_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Tạo bộ benchmark chuẩn hóa VietLawBench từ Neo4j")
    parser.add_argument("--single", type=int, default=600)
    parser.add_argument("--multi", type=int, default=400)
    args = parser.parse_args()

    builder = VietLawBenchBuilder()
    try:
        builder.build_and_export_all(args.single, args.multi)
    finally:
        builder.close()


if __name__ == "__main__":
    main()