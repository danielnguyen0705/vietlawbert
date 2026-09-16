"""
ingest_pipeline.py - Luồng trung gian (Decoupled ETL Worker) xử lý từ Shard thô sang CSDL lai.
Tối ưu hóa tài nguyên: Khống chế CPU Threads, Doc-level Resume, UUID RFC 4122 và Safe Attribute Parsing.
"""

from __future__ import annotations

import os
import gc
import json
import gzip
import logging
import argparse
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from bs4 import BeautifulSoup

import torch
try:
    # Khóa cứng luồng CPU tránh gây nghẽn giao diện người dùng Dell G7
    torch.set_num_threads(2)
except Exception:
    pass

from sentence_transformers import SentenceTransformer

from configs.config import config
from configs.paths import RAW_SHARDS_DIR
from preprocess.ast_parser import HybridASTParser
from preprocess.validity import is_currently_effective
from database.qdrant_client import QdrantClientWrapper
from rag.es_retriever import LegalElasticsearchRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s | [%(levelname)s] | %(message)s")
logger = logging.getLogger("VietLawBERT_IngestPipeline")

CHECKPOINT_FILE = config.STORAGE_ROOT / ".ingest_checkpoint.json"


def load_checkpoint() -> Set[str]:
    """Tải danh sách các tệp Shard đã hoàn tất 100% từ đĩa cứng."""
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            return set()
    return set()


def save_checkpoint(completed_shards: Set[str]) -> None:
    """Ghi vết checkpoint nguyên tử chống hỏng dữ liệu khi sập nguồn."""
    try:
        temp_file = CHECKPOINT_FILE.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(list(completed_shards), f, ensure_ascii=False, indent=2)
        temp_file.replace(CHECKPOINT_FILE)
    except Exception as exc:
        logger.warning("Không thể ghi checkpoint: %s", exc)


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
        skip_existing: bool = True,
    ):
        self.qdrant_host = qdrant_host or config.QDRANT_HOST
        self.qdrant_port = qdrant_port or config.QDRANT_PORT
        self.es_host = es_host or config.ES_HOST
        self.model_name = model_name_or_path or config.BASE_MODEL_NAME
        self.vector_dim = vector_dim or config.QDRANT_VECTOR_DIM
        self.batch_size = batch_size or config.EMBED_BATCH_SIZE
        self.skip_existing = skip_existing

        self.parser = HybridASTParser()
        self.qdrant = QdrantClientWrapper(host=self.qdrant_host, port=self.qdrant_port)
        try:
            self.qdrant.init_collection(collection_name=config.QDRANT_COLLECTION_NAME, vector_dim=self.vector_dim)
        except Exception as init_err:
            logger.warning("Sử dụng collection Qdrant hiện có: %s", init_err)

        self.es = LegalElasticsearchRetriever(hosts=[self.es_host], index_name=config.ES_INDEX_NAME)

        self.device = device or config.EMBED_DEVICE
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
                del soup
                if len(clean_text) > 50:
                    return clean_text
            except Exception as exc:
                logger.debug("Lỗi giải mã HTML thô: %s", exc)

        html_path = doc_record.get("html_path")
        if html_path and Path(html_path).exists():
            try:
                with open(html_path, "r", encoding="utf-8", errors="ignore") as hf:
                    raw_from_file = hf.read()
                    if raw_from_file:
                        soup = BeautifulSoup(raw_from_file, "html.parser")
                        for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "iframe"]):
                            tag.decompose()
                        clean_text = soup.get_text("\n", strip=True)
                        del soup
                        if len(clean_text) > 50:
                            return clean_text
            except Exception as exc:
                logger.debug("Lỗi đọc tệp html_path %s: %s", html_path, exc)

        return ""

    @staticmethod
    def _parse_safe_status(doc_record: Dict[str, Any], meta_detail: Dict[str, Any], meta_api: Dict[str, Any]) -> str:
        """Trích xuất trạng thái an toàn chống lỗi AttributeError khi effStatus là integer."""
        if doc_record.get("status"):
            return str(doc_record["status"])

        eff_detail = meta_detail.get("effStatus")
        if isinstance(eff_detail, dict):
            return str(eff_detail.get("name") or "Chưa xác định")
        if isinstance(eff_detail, (int, str)) and str(eff_detail).strip():
            return "Còn hiệu lực" if str(eff_detail) == "1" else "Hết hiệu lực"

        eff_api = meta_api.get("effStatus")
        if isinstance(eff_api, dict):
            return str(eff_api.get("name") or "Chưa xác định")
        if isinstance(eff_api, (int, str)) and str(eff_api).strip():
            return "Còn hiệu lực" if str(eff_api) == "1" else "Hết hiệu lực"

        return "Chưa xác định"

    def process_raw_shard(self, shard_file_path: str | Path) -> int:
        path = Path(shard_file_path)
        if not path.exists():
            raise FileNotFoundError(path)

        logger.info("Xử lý Shard dữ liệu: %s...", path.name)
        is_gzip = "".join(path.suffixes).endswith(".gz") or path.name.endswith(".gz")

        chunk_buffer: List[Dict[str, Any]] = []
        total_chunks_in_shard = 0
        docs_count = 0
        skipped_existing_docs = 0

        try:
            opener = gzip.open(path, "rt", encoding="utf-8") if is_gzip else open(path, "r", encoding="utf-8")
            with opener as f:
                for line_idx, line in enumerate(f, start=1):
                    clean_line = line.strip()
                    if not clean_line:
                        continue

                    try:
                        doc_record = json.loads(clean_line)
                    except json.JSONDecodeError:
                        continue

                    doc_id = str(doc_record.get("doc_id") or doc_record.get("item_id") or f"doc_{line_idx}")

                    # CƠ CHẾ RESUME CẤP VĂN BẢN: Bỏ qua văn bản đã tồn tại trong Qdrant để cứu thời gian CPU
                    if self.skip_existing and self.qdrant.doc_exists(doc_id):
                        skipped_existing_docs += 1
                        continue

                    raw_text = self._extract_clean_text_fallback(doc_record)
                    if not raw_text:
                        continue

                    meta_detail = doc_record.get("metadata_detail") or {}
                    meta_api = doc_record.get("metadata_api") or {}

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
                    status_raw = self._parse_safe_status(doc_record, meta_detail, meta_api)
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

                    chunk_occurrences: Dict[str, int] = {}
                    for ch in ast_chunks:
                        ch_id = str(ch.get("chunk_id") or ch.get("metadata", {}).get("chunk_id") or str(uuid.uuid4()))
                        # Amending documents can quote several articles with the
                        # same number. Preserve every segment with stable IDs.
                        chunk_occurrences[ch_id] = chunk_occurrences.get(ch_id, 0) + 1
                        occurrence = chunk_occurrences[ch_id]
                        if occurrence > 1:
                            ch_id = f"{ch_id}_occurrence_{occurrence}"
                        ch["chunk_id"] = ch_id
                        ch["doc_id"] = doc_id
                        ch["doc_number"] = doc_number

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

        except Exception as read_err:
            logger.error("Shard %s thất bại; không ghi checkpoint: %s", path.name, read_err)
            raise

        gc.collect()
        logger.info(
            "✓ Hoàn tất Shard %s: Đã nạp %d văn bản mới (%d chunks) | Bỏ qua %d văn bản đã có sẵn.",
            path.name,
            docs_count,
            total_chunks_in_shard,
            skipped_existing_docs,
        )
        return total_chunks_in_shard

    def _flush_chunks_to_storage(self, chunks: List[Dict[str, Any]]) -> None:
        if not chunks:
            return

        texts = [c.get("text") or c.get("content") or "" for c in chunks]

        with torch.inference_mode():
            embeddings = self.encoder.encode(
                texts,
                batch_size=min(len(texts), self.batch_size),
                show_progress_bar=False,
                normalize_embeddings=True,
            )

        qdrant_payloads = []
        for i, chunk in enumerate(chunks):
            chunk_copy = dict(chunk)
            chunk_copy["embedding"] = embeddings[i][: self.vector_dim].tolist()

            chunk_copy["is_effective"] = is_currently_effective(chunk.get("metadata", {}))

            chunk_content = chunk.get("content") or chunk.get("text", "")
            chunk_copy["content"] = chunk_content
            chunk_copy["text"] = chunk_content
            chunk["is_effective"] = chunk_copy["is_effective"]
            chunk["content"] = chunk_content
            chunk["text"] = chunk_content
            qdrant_payloads.append(chunk_copy)

        # 1. Nạp Qdrant - Cơ chế Fail-Fast bảo vệ tính toàn vẹn
        upserted = self.qdrant.upsert_batch(
            records=qdrant_payloads,
            collection_name=config.QDRANT_COLLECTION_NAME,
        )
        if upserted != len(qdrant_payloads):
            raise RuntimeError("Qdrant nạp thất bại toàn bộ batch. Dừng tiến trình để tránh lệch pha dữ liệu!")

        # 2. Nạp Elasticsearch
        try:
            if self.es.bulk_index_chunks(chunks) != len(chunks):
                raise RuntimeError("Elasticsearch không nạp đủ batch")
        except Exception as es_err:
            logger.error("Ngoại lệ khi nạp Elasticsearch: %s", es_err)
            raise es_err

        del texts, embeddings, qdrant_payloads
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Chạy luồng trung gian nạp dữ liệu từ Shards vào CSDL")
    parser.add_argument("--shard-path", default=str(RAW_SHARDS_DIR), help="Đường dẫn thư mục Shard")
    parser.add_argument("--model-name", default=config.BASE_MODEL_NAME, help="Tên backbone encoder")
    parser.add_argument("--dim", type=int, default=config.QDRANT_VECTOR_DIM, help="Số chiều vector (256)")
    parser.add_argument("--batch-size", type=int, default=config.EMBED_BATCH_SIZE, help="Kích thước mini-batch")
    parser.add_argument("--device", default=config.EMBED_DEVICE, help="Thiết bị tính toán (cpu/cuda)")
    args = parser.parse_args()

    worker = IngestPipelineWorker(
        model_name_or_path=args.model_name,
        vector_dim=args.dim,
        batch_size=args.batch_size,
        device=args.device,
    )

    completed_shards = load_checkpoint()
    p = Path(args.shard_path)

    if p.is_dir():
        all_candidates = sorted(list(p.glob("*.jsonl*")))
        files = [
            f for f in all_candidates
            if (f.name.endswith(".jsonl.gz") or f.name.endswith(".jsonl"))
            and not f.name.endswith(".corrupted")
            and not f.name.endswith(".quarantine.jsonl")
            and not f.name.endswith(".tmp")
        ]
        logger.info("Tìm thấy %d shards hợp lệ trong %s (Đã nạp trước đó: %d).", len(files), p, len(completed_shards))
        for f in files:
            if f.name in completed_shards:
                logger.info("[CHECKPOINT SKIP] Bỏ qua Shard đã nạp thành công: %s", f.name)
                continue
            worker.process_raw_shard(f)
            completed_shards.add(f.name)
            save_checkpoint(completed_shards)
    else:
        worker.process_raw_shard(p)


if __name__ == "__main__":
    main()
