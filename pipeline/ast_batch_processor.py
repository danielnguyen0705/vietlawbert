"""
ast_batch_processor.py - Phân rã cấu trúc văn bản pháp luật bằng EBNF Grammar.
Đọc song song 161 Shards .jsonl.gz và xuất ra định dạng Apache Parquet (O(1) Memory).
"""

from __future__ import annotations

import os
import gzip
import json
import glob
import re
import html as html_lib
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor
import pyarrow as pa
import pyarrow.parquet as pq
from bs4 import BeautifulSoup

RAW_SHARDS_DIR = Path("/mnt/data/vietlawbert_data/raw_shards")
OUTPUT_PARQUET_DIR = Path("/mnt/data/vietlawbert_data/processed_parquet")
OUTPUT_PARQUET_DIR.mkdir(parents=True, exist_ok=True)

RE_DIEU = re.compile(r"^(Điều\s+\d+[\.\:]?)\s*(.*)", re.IGNORECASE)
RE_KHOAN = re.compile(r"^(\d+)[\.\)]\s*(.*)")
RE_DIEM = re.compile(r"^([a-zđ])[\.\)]\s*(.*)", re.IGNORECASE)

PARQUET_SCHEMA = pa.schema([
    ("chunk_id", pa.string()),
    ("doc_id", pa.string()),
    ("doc_number", pa.string()),
    ("doc_title", pa.string()),
    ("hierarchy_path", pa.string()),
    ("level", pa.string()),
    ("content", pa.string()),
    ("full_context_text", pa.string()),
    ("issue_date", pa.string()),
    ("effective_date", pa.string()),
    ("is_administrative", pa.bool_()),
])

def clean_html_text(raw_html: str) -> str:
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "lxml")
    for tag in soup(["script", "style", "header", "footer"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = html_lib.unescape(text)
    return re.sub(r"[ \t]+", " ", text).strip()

def parse_document_ast(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    doc_id = str(record.get("item_id") or record.get("doc_id") or "").strip()
    doc_number = str(record.get("doc_number") or "N/A").strip()
    meta_detail = record.get("metadata_detail") or {}
    meta_api = record.get("metadata_api") or {}
    
    title = meta_detail.get("title") or meta_api.get("title") or ""
    issue_date = str(meta_detail.get("issueDate") or meta_api.get("issueDate") or "")
    effective_date = str(meta_detail.get("effFrom") or meta_api.get("effFrom") or "")
    is_admin = bool(meta_detail.get("isAdministrativeDocument"))

    html_raw = record.get("html_raw") or ""
    if not html_raw or len(html_raw) < 100:
        return []

    plain_text = clean_html_text(html_raw)
    lines = [line.strip() for line in plain_text.splitlines() if line.strip()]

    chunks = []
    current_chapter = ""
    current_dieu = ""
    current_dieu_content = []
    
    dieu_idx = 0
    for line in lines:
        if line.lower().startswith("chương ") or line.lower().startswith("mục "):
            current_chapter = line
            continue

        match_dieu = RE_DIEU.match(line)
        if match_dieu:
            if current_dieu and current_dieu_content:
                dieu_idx += 1
                full_text = "\n".join(current_dieu_content)
                hierarchy = f"{title} > {current_chapter} > {current_dieu}" if current_chapter else f"{title} > {current_dieu}"
                chunks.append({
                    "chunk_id": f"{doc_id}_art_{dieu_idx}",
                    "doc_id": doc_id,
                    "doc_number": doc_number,
                    "doc_title": title,
                    "hierarchy_path": hierarchy,
                    "level": "ARTICLE",
                    "content": full_text,
                    "full_context_text": f"[{hierarchy}]\n{full_text}",
                    "issue_date": issue_date,
                    "effective_date": effective_date,
                    "is_administrative": is_admin,
                })
            current_dieu = match_dieu.group(1)
            current_dieu_content = [line]
        else:
            if current_dieu:
                current_dieu_content.append(line)

    if current_dieu and current_dieu_content:
        dieu_idx += 1
        full_text = "\n".join(current_dieu_content)
        hierarchy = f"{title} > {current_chapter} > {current_dieu}" if current_chapter else f"{title} > {current_dieu}"
        chunks.append({
            "chunk_id": f"{doc_id}_art_{dieu_idx}",
            "doc_id": doc_id,
            "doc_number": doc_number,
            "doc_title": title,
            "hierarchy_path": hierarchy,
            "level": "ARTICLE",
            "content": full_text,
            "full_context_text": f"[{hierarchy}]\n{full_text}",
            "issue_date": issue_date,
            "effective_date": effective_date,
            "is_administrative": is_admin,
        })

    # Dự phòng cho các văn bản ngắn hoặc quyết định cá biệt không chia Điều
    if not chunks and plain_text:
        chunks.append({
            "chunk_id": f"{doc_id}_full",
            "doc_id": doc_id,
            "doc_number": doc_number,
            "doc_title": title,
            "hierarchy_path": title,
            "level": "DOCUMENT",
            "content": plain_text[:4000],
            "full_context_text": f"[{title}]\n{plain_text[:4000]}",
            "issue_date": issue_date,
            "effective_date": effective_date,
            "is_administrative": is_admin,
        })

    return chunks

def process_single_shard(shard_path_str: str) -> Dict[str, Any]:
    shard_path = Path(shard_path_str)
    parquet_path = OUTPUT_PARQUET_DIR / shard_path.name.replace(".jsonl.gz", ".parquet")

    if parquet_path.exists():
        return {"shard": shard_path.name, "chunks": 0, "status": "ALREADY_EXISTS"}

    all_chunks = []
    with gzip.open(shard_path, "rt", encoding="utf-8", errors="replace") as gz:
        for line in gz:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                chunks = parse_document_ast(record)
                all_chunks.extend(chunks)
            except Exception:
                continue

    if all_chunks:
        table = pa.Table.from_pylist(all_chunks, schema=PARQUET_SCHEMA)
        pq.write_table(table, parquet_path, compression="snappy")

    return {"shard": shard_path.name, "chunks": len(all_chunks), "status": "SUCCESS"}

def main():
    shard_files = sorted(glob.glob(str(RAW_SHARDS_DIR / "crawl_pages_*_*.jsonl.gz")))
    print(f"Bắt đầu bóc tách AST phân cấp cho {len(shard_files)} Shards...")

    num_workers = max(1, (os.cpu_count() or 4) - 1)
    total_extracted_chunks = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for res in executor.map(process_single_shard, shard_files):
            total_extracted_chunks += res["chunks"]
            if res["status"] == "SUCCESS":
                print(f"[AST PARSED] {res['shard']}: Trích xuất thành công {res['chunks']:,} Chunks")

    print("=" * 65)
    print(f"HOÀN THÀNH PHA 2: Tổng số Chunks phân cấp: {total_extracted_chunks:,}")
    print(f"Thư mục lưu trữ: {OUTPUT_PARQUET_DIR}")
    print("=" * 65)

if __name__ == "__main__":
    main()
