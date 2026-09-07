"""
build_benchmark.py - Động cơ tự động sinh tập dữ liệu kiểm chuẩn VietLawBench (Single-hop & Multi-hop).
Khai thác đồ thị tri thức Neo4j và cấu trúc cây phân cấp AST để tạo Ground-Truth chuẩn cho ACL/EMNLP.
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from neo4j import GraphDatabase

from configs.paths import ROOT_DIR, DATA_STORAGE_ROOT, BENCHMARK_DIR
from configs.config import config
from configs.logging_config import get_subsystem_logger
from artifacts.canonical import read_jsonl, write_jsonl

logger = get_subsystem_logger("benchmark", "build_benchmark")


def extract_key_phrase(text: str) -> str:
    """Rút trích cụm chủ đề cốt lõi từ văn bản điều luật."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    first_line = lines[0] if lines else text[:100]
    # Bỏ tiền tố số thứ tự (VD: "1. Người điều khiển xe..." -> "Người điều khiển xe...")
    clean = re_prefix = re_sub = re_clean = re_replace = re = None
    import re
    first_line = re.sub(r"^(\d+|[a-zđ])[\.\)\s\-]+", "", first_line, flags=re.IGNORECASE).strip()
    return first_line[:120]


class VietLawBenchBuilder:
    """Bộ điều phối tự động tạo câu hỏi trắc nghiệm và tình huống pháp lý kiểm chuẩn."""

    def __init__(self, neo4j_uri: Optional[str] = None):
        self.neo4j_uri = neo4j_uri or getattr(config, "NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = getattr(config, "NEO4J_USER", "neo4j")
        self.neo4j_password = getattr(config, "NEO4J_PASSWORD", "vietlawbert")

        logger.info(f"Kết nối Neo4j tại {self.neo4j_uri} để trích xuất topo phân cấp...")
        self.driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_password)
        )
        self.benchmark_dir = Path(BENCHMARK_DIR)
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)

    def close(self):
        self.driver.close()

    def generate_single_hop_bench(self, sample_size: int = 500) -> List[Dict[str, Any]]:
        """
        Sinh tập kiểm thử Single-hop:
        Mỗi câu hỏi truy vấn trực tiếp vào nội dung cốt lõi của 1 Điều luật cụ thể.
        """
        logger.info(f"Đang sinh {sample_size:,} mẫu Single-hop từ Neo4j...")
        cypher = """
        MATCH (doc:LawDocument)-[:HAS_CHAPTER]->(ch:Chapter)-[:HAS_ARTICLE]->(art:Article)-[:HAS_CHUNK]->(ck:Chunk)
        WHERE doc.doc_number IS NOT NULL 
          AND doc.doc_number <> 'N/A'
          AND art.name STARTS WITH 'Điều'
          AND size(ck.text) > 100
        WITH doc, art, ck, rand() AS r
        ORDER BY r
        LIMIT $limit
        RETURN doc.doc_number AS doc_number,
               doc.name AS doc_name,
               art.name AS article,
               ck.chunk_id AS chunk_id,
               ck.text AS content
        """
        samples = []
        templates = [
            "Theo quy định tại {article} {doc_number}, {topic} được quy định như thế nào?",
            "Căn cứ vào {article} của {doc_number}, nội dung điều chỉnh về {topic} là gì?",
            "Quy định pháp lý về {topic} được nêu tại {article} {doc_number} như thế nào?",
        ]

        with self.driver.session() as session:
            records = session.run(cypher, parameters={"limit": sample_size * 2}).data()

        seen_targets = set()
        for rec in records:
            doc_num = str(rec["doc_number"]).strip()
            article = str(rec["article"]).strip()
            content = str(rec["content"]).strip()
            target_key = (doc_num, article)

            if target_key in seen_targets:
                continue

            topic = extract_key_phrase(content)
            if len(topic) < 15:
                continue

            tpl = random.choice(templates)
            query = tpl.format(article=article, doc_number=doc_num, topic=topic)

            samples.append({
                "benchmark_id": f"single_hop_{len(samples)+1:05d}",
                "query": query,
                "ground_truth_doc_number": doc_num,
                "ground_truth_article": article,
                "target_chunk_id": rec["chunk_id"],
                "evidence_text": content[:300],
            })
            seen_targets.add(target_key)
            if len(samples) >= sample_size:
                break

        logger.info(f"✓ Hoàn thành sinh Single-hop: {len(samples)} mẫu chuẩn.")
        return samples

    def generate_multi_hop_bench(self, sample_size: int = 500) -> List[Dict[str, Any]]:
        """
        Sinh tập kiểm thử Multi-hop:
        Dựa trên cạnh quan hệ dẫn chiếu thực tế (DAN_CHIEU, SUA_DOI_BO_SUNG, CAN_CU_BAN_HANH).
        Yêu cầu mô hình phải thu hồi chính xác cả 2 căn cứ pháp luật ở 2 văn bản khác nhau.
        """
        logger.info(f"Đang sinh {sample_size:,} mẫu Multi-hop từ các liên kết dẫn chiếu Neo4j...")
        cypher = """
        MATCH (doc1:LawDocument)-[r:DAN_CHIEU|SUA_DOI_BO_SUNG|CAN_CU_BAN_HANH|QUY_DINH_CHI_TIET_HUONG_DAN]->(doc2:LawDocument)
        WHERE doc1.doc_number IS NOT NULL AND doc1.doc_number <> 'N/A'
          AND doc2.doc_number IS NOT NULL AND doc2.doc_number <> 'N/A'
          AND doc1.doc_id <> doc2.doc_id
        MATCH (doc1)-[:HAS_CHAPTER]->()-[:HAS_ARTICLE]->(art1:Article)-[:HAS_CHUNK]->(ck1:Chunk)
        MATCH (doc2)-[:HAS_CHAPTER]->()-[:HAS_ARTICLE]->(art2:Article)-[:HAS_CHUNK]->(ck2:Chunk)
        WHERE size(ck1.text) > 80 AND size(ck2.text) > 80
        WITH doc1, art1, ck1, r, doc2, art2, ck2, rand() AS rand_val
        ORDER BY rand_val
        LIMIT $limit
        RETURN doc1.doc_number AS doc1_num,
               art1.name AS art1_name,
               ck1.text AS ck1_text,
               type(r) AS rel_type,
               doc2.doc_number AS doc2_num,
               art2.name AS art2_name,
               ck2.text AS ck2_text
        """
        samples = []
        multi_templates = [
            "Khi áp dụng {art1} {doc1_num} có liên quan đến việc {rel_desc} theo {art2} {doc2_num}, quy định phối hợp này như thế nào?",
            "Căn cứ vào {doc1_num} ({art1}) và các quy định bổ trợ tại {doc2_num} ({art2}), việc xử lý tình huống được thực hiện ra sao?",
        ]
        rel_desc_map = {
            "DAN_CHIEU": "dẫn chiếu áp dụng",
            "SUA_DOI_BO_SUNG": "sửa đổi, bổ sung",
            "CAN_CU_BAN_HANH": "căn cứ ban hành thẩm quyền",
            "QUY_DINH_CHI_TIET_HUONG_DAN": "hướng dẫn chi tiết thi hành",
        }

        with self.driver.session() as session:
            records = session.run(cypher, parameters={"limit": sample_size * 3}).data()

        seen_pairs = set()
        for rec in records:
            d1_num = str(rec["doc1_num"]).strip()
            a1_name = str(rec["art1_name"]).strip()
            d2_num = str(rec["doc2_num"]).strip()
            a2_name = str(rec["art2_name"]).strip()
            pair_key = (d1_num, a1_name, d2_num, a2_name)

            if pair_key in seen_pairs:
                continue

            rel_type = rec["rel_type"]
            rel_desc = rel_desc_map.get(rel_type, "liên quan thi hành")
            tpl = random.choice(multi_templates)
            query = tpl.format(art1=a1_name, doc1_num=d1_num, rel_desc=rel_desc, art2=a2_name, doc2_num=d2_num)

            samples.append({
                "benchmark_id": f"multi_hop_{len(samples)+1:05d}",
                "query": query,
                "ground_truth_docs": [
                    {"doc_number": d1_num, "article": a1_name},
                    {"doc_number": d2_num, "article": a2_name},
                ],
                "relation_link": rel_type,
            })
            seen_pairs.add(pair_key)
            if len(samples) >= sample_size:
                break

        logger.info(f"✓ Hoàn thành sinh Multi-hop: {len(samples)} mẫu chuẩn.")
        return samples

    def build_all(self, single_size: int = 500, multi_size: int = 500):
        single_samples = self.generate_single_hop_bench(single_size)
        multi_samples = self.generate_multi_hop_bench(multi_size)

        single_path = self.benchmark_dir / "single_hop.jsonl"
        multi_path = self.benchmark_dir / "multi_hop.jsonl"

        write_jsonl(single_path, single_samples)
        write_jsonl(multi_path, multi_samples)

        logger.info(f"✓ Đã ghi nhận tập kiểm chuẩn Single-hop: {single_path.resolve()}")
        logger.info(f"✓ Đã ghi nhận tập kiểm chuẩn Multi-hop: {multi_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Tạo bộ benchmark đánh giá VietLawBench từ Neo4j")
    parser.add_argument("--single-size", type=int, default=500, help="Số lượng mẫu Single-hop")
    parser.add_argument("--multi-size", type=int, default=500, help="Số lượng mẫu Multi-hop")
    args = parser.parse_args()

    builder = VietLawBenchBuilder()
    try:
        builder.build_all(single_size=args.single_size, multi_size=args.multi_size)
    finally:
        builder.close()


if __name__ == "__main__":
    main()
