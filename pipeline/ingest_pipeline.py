"""
ingest_pipeline.py - Luồng trung gian (Decoupled ETL Worker) xử lý từ Shard thô sang CSDL lai.
Tối ưu hóa tài nguyên phần cứng yếu: Khống chế CPU Threads, mini-batch nhỏ và giải phóng RAM chủ động.
"""

from __future__ import annotations

import os
import gc
import json
import gzip
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup

import torch
from sentence_transformers import SentenceTransformer

from configs.config import config
from preprocess.ast_parser import HybridASTParser
from database.qdrant_client import QdrantClientWrapper
from rag.es_retriever import LegalElasticsearchRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s | [%(levelname)s] | %(message)s")
logger = logging.getLogger("VietLawBERT_IngestPipeline")

# Khóa trần số nhân CPU của PyTorch để nhường tài nguyên cho HĐH và giao diện người dùng
torch.set_num_threads(2)
torch.set_num_interop_threads(1)


class IngestPipelineWorker:
    def __init__(
        self,
        qdrant_host: Optional[str] = None,
        qdrant_port: Optional[int] = None,
        es_host: Optional[str] = None,
        model_name_or_path: Optional[str] = None,
        vector_dim: Optional[int] = None,
        batch_size: Optional[int] = None,
        device: Optional[str] = None,
    ):
        self.qdrant_host = qdrant_host or config.QDRANT_HOST
        self.qdrant_port = qdrant_port or config.QDRANT_PORT
        self.es_host = es_host or config.ES_HOST
        self.model_name = model_name_or_path or config.BASE_MODEL_NAME
        self.vector_dim = vector_dim or config.QDRANT_VECTOR_DIM
        self.batch_size = batch_size or 8  # Mặc định kích thước nhỏ cho máy yếu

        self.parser = HybridASTParser()
        self.qdrant = QdrantClientWrapper(host=self.qdrant_host, port=self.qdrant_port)
        self.qdrant.init_collection(collection_name=config.QDRANT_COLLECTION_NAME, vector_dim=self.vector_dim)
        self.es = LegalElasticsearchRetriever(hosts=[self.es_host], index_name=config.ES_INDEX_NAME)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Đang nạp mô hình %s trên [%s] (PyTorch threads=2)...", self.model_name, self.device)
        self.encoder = SentenceTransformer(self.model_name, device=self.device)

    def _extract_clean_text_fallback(self, doc_record: Dict[str, Any]) -> str:
        text = doc_record.get("full_text") or doc_record.get("text") or ""
        if text and len(str(text).strip()) > 50:
            return str(text).strip()

        html_raw = doc_record.get("html_raw") or ""
        if html_raw:
            try:
                soup = BeautifulSoup(html_raw, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "iframe"]):
                    tag.decompose()
                clean_text = soup.get_text("\n", strip=True)
                if len(clean_text) > 50:
                    return clean_text
            except Exception as exc:
                logger.debug("Lỗi giải mã HTML thô: %s", exc)

        return ""

    def process_raw_shard(self, shard_file_path: str | Path) -> int:
        path = Path(shard_file_path)
        if not path.exists():
            logger.error("Không tìm thấy tệp shard: %s", path)
            return 0

        logger.info("Xử lý Shard dữ liệu: %s...", path.name)
        opener = gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else open(path, "r", encoding="utf-8")

        chunk_buffer: List[Dict[str, Any]] = []
        total_chunks_in_shard = 0
        docs_count = 0

        with opener as f:
            for line_idx, line in enumerate(f, start=1):
                clean_line = line.strip()
                if not clean_line:
                    continue

                try:
                    doc_record = json.loads(clean_line)
                except json.JSONDecodeError:
                    continue

                raw_text = self._extract_clean_text_fallback(doc_record)
                if not raw_text:
                    continue

                meta_detail = doc_record.get("metadata_detail") or {}
                meta_api = doc_record.get("metadata_api") or {}

                doc_id = str(doc_record.get("doc_id") or doc_record.get("item_id") or f"doc_{line_idx}")
                doc_number = str(
                    doc_record.get("doc_number")
                    or meta_detail.get("docNum")
                    or meta_api.get("docNum")
                    or "N/A"
                )
                title = str(
                    doc_record.get("title")
                    or meta_detail.get("title")
                    or meta_api.get("title")
                    or "Văn bản pháp luật"
                )
                effective_date = str(
                    doc_record.get("effective_date")
                    or meta_detail.get("effFrom")
                    or meta_api.get("effFrom")
                    or "Chưa xác định"
                )
                status_raw = str(
                    doc_record.get("status")
                    or (meta_detail.get("effStatus") or {}).get("name")
                    or (meta_api.get("effStatus") or {}).get("name")
                    or "Còn hiệu lực"
                )
                org = str(
                    doc_record.get("co_quan")
                    or meta_detail.get("agencyName")
                    or meta_api.get("agencyName")
                    or "N/A"
                )

                metadata = {
                    "doc_id": doc_id,
                    "doc_number": doc_number,
                    "title": title,
                    "effective_date": effective_date,
                    "status": status_raw,
                    "co_quan": org,
                }

                try:
                    ast_chunks = self.parser.parse_document(raw_text, metadata)
                except Exception as parse_err:
                    logger.warning("Bỏ qua lỗi AST doc %s: %s", doc_id, parse_err)
                    continue

                chunk_buffer.extend(ast_chunks)
                docs_count += 1

                if len(chunk_buffer) >= self.batch_size:
                    self._flush_chunks_to_storage(chunk_buffer)
                    total_chunks_in_shard += len(chunk_buffer)
                    chunk_buffer.clear()

        if chunk_buffer:
            self._flush_chunks_to_storage(chunk_buffer)
            total_chunks_in_shard += len(chunk_buffer)
            chunk_buffer.clear()

        # Dọn dẹp bộ nhớ RAM sau khi hoàn tất toàn bộ shard
        gc.collect()

        logger.info(
            "✓ Đã nạp thành công Shard %s: %d văn bản -> %d chunks vào Qdrant & ES.",
            path.name,
            docs_count,
            total_chunks_in_shard,
        )
        return total_chunks_in_shard

    def _flush_chunks_to_storage(self, chunks: List[Dict[str, Any]]) -> None:
        if not chunks:
            return

        texts = [c["text"] for c in chunks]

        # Chạy inference_mode triệt tiêu computation graph giải phóng VRAM/RAM
        with torch.inference_mode():
            embeddings = self.encoder.encode(
                texts,
                batch_size=min(len(texts), 8),
                show_progress_bar=False,
                normalize_embeddings=True,
            )

        qdrant_payloads = []
        for i, chunk in enumerate(chunks):
            chunk_copy = dict(chunk)
            chunk_copy["embedding"] = embeddings[i][: self.vector_dim].tolist()

            status_str = str(chunk.get("metadata", {}).get("status", ""))
            chunk_copy["is_effective"] = ("hết hiệu lực" not in status_str.lower())

            chunk_copy["content"] = chunk.get("text", "")
            chunk["is_effective"] = chunk_copy["is_effective"]
            chunk["content"] = chunk_copy["content"]
            qdrant_payloads.append(chunk_copy)

        self.qdrant.upsert_batch(
            records=qdrant_payloads,
            collection_name=config.QDRANT_COLLECTION_NAME,
        )
        self.es.bulk_index_chunks(chunks)

        del texts, embeddings, qdrant_payloads
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Chạy luồng trung gian nạp dữ liệu từ Shards vào CSDL")
    parser.add_argument("--shard-path", default=str(config.STORAGE_ROOT / "raw_shards"), help="Đường dẫn thư mục Shard")
    parser.add_argument("--model-name", default=config.BASE_MODEL_NAME, help="Tên backbone encoder")
    parser.add_argument("--dim", type=int, default=config.QDRANT_VECTOR_DIM, help="Số chiều vector (256)")
    parser.add_argument("--batch-size", type=int, default=8, help="Kích thước mini-batch nhỏ (khuyến nghị 8)")
    parser.add_argument("--device", default="cpu", help="Thiết bị tính toán (cpu/cuda)")
    args = parser.parse_args()

    worker = IngestPipelineWorker(
        model_name_or_path=args.model_name,
        vector_dim=args.dim,
        batch_size=args.batch_size,
        device=args.device,
    )

    p = Path(args.shard_path)
    if p.is_dir():
        files = sorted(list(p.glob("*.jsonl*")))
        logger.info("Tìm thấy %d shards trong thư mục %s.", len(files), p)
        for f in files:
            worker.process_raw_shard(f)
    else:
        worker.process_raw_shard(p)


if __name__ == "__main__":
    main()