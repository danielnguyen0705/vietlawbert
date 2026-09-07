"""
contextualizer.py - Module làm giàu ngữ cảnh (Context Enrichment) cho từng đoạn trích pháp luật.
Hỗ trợ Deterministic Structural Injection tốc độ cao, LLM-based Summary Injection và tương thích CLI.
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any

from openai import OpenAI
from confluent_kafka import Producer

from configs.paths import DATA_STORAGE_ROOT, RAW_SHARDS_DIR, ARTIFACTS_DIR
from configs.config import config
from configs.logging_config import get_subsystem_logger
from artifacts.canonical import read_jsonl
from preprocess.legal_chunker import chunk_legal_document, LegalChunk, extract_cross_references
from preprocess.text_cleaner import clean_boilerplate, extract_doc_type, extract_doc_number, extract_effective_date

logger = get_subsystem_logger("preprocess", "contextualizer")

_llm_client: Optional[OpenAI] = None
_llm_warning_logged = False


def get_llm_client() -> OpenAI:
    global _llm_client
    if _llm_client is None:
        _llm_client = OpenAI(
            base_url=getattr(config, "LLM_API_BASE", "http://localhost:11434/v1"),
            api_key=getattr(config, "LLM_API_KEY", "ollama"),
            timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "30.0")),
            max_retries=1,
        )
    return _llm_client


SYSTEM_PROMPT = """Bạn là chuyên gia NLP pháp lý cao cấp, chuyên phân tích văn bản quy phạm pháp luật Việt Nam.
NHIỆM VỤ: Viết đúng 1-2 câu giải thích ngắn gọn làm rõ ngữ cảnh cho đoạn trích luật.
CẤU TRÚC PHẢN HỒI:
"Quy định này điều chỉnh về [HÀNH VI/CHẾ TÀI] áp dụng đối với [ĐỐI TƯỢNG], thuộc [LOẠI VĂN BẢN] số [SỐ HIỆU]. [NỘI DUNG CHÍNH]."
"""


def call_llm(prompt: str, model: Optional[str] = None, max_tokens: int = 150) -> str:
    """Gọi LLM Server để tạo ngữ cảnh tóm lược."""
    global _llm_warning_logged
    client = get_llm_client()
    target_model = model or getattr(config, "CONTEXTUALIZER_MODEL", "qwen2.5:1.5b")

    try:
        response = client.chat.completions.create(
            model=target_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        if not _llm_warning_logged:
            logger.warning("Dịch vụ LLM không phản hồi (%s). Chuyển sang dùng ngữ cảnh cấu trúc.", exc)
            _llm_warning_logged = True
        return ""


# Alias tương thích ngược cho các CLI cũ
call_ollama = call_llm


def build_deterministic_context(chunk: LegalChunk, doc_number: str, doc_type: str, effective_date: str) -> str:
    """
    Tiêm ngữ cảnh cấu trúc xác định (Deterministic Situational Injection).
    Khắc phục độ trễ LLM, đạt tốc độ hàng chục nghìn chunk/giây.
    """
    hierarchy = chunk.hierarchy
    hierarchy_parts = [
        val for val in [
            hierarchy.phan, hierarchy.chuong, hierarchy.muc,
            hierarchy.tieu_muc, hierarchy.dieu, hierarchy.khoan, hierarchy.diem
        ] if val
    ]
    hierarchy_str = " > ".join(hierarchy_parts) if hierarchy_parts else "Toàn văn"
    
    header = f"[Căn cứ: {doc_type} số {doc_number} | Hiệu lực: {effective_date} | Phân vị: {hierarchy_str}]"
    return f"{header}\n{chunk.text}"


def build_prompt(chunk: LegalChunk, doc_number: str, doc_type: str, effective_date: str) -> str:
    """Xây dựng Prompt cung cấp cây phả hệ pháp luật cho LLM."""
    hierarchy = chunk.hierarchy
    hierarchy_parts = [
        val for val in [
            hierarchy.phan, hierarchy.chuong, hierarchy.muc,
            hierarchy.tieu_muc, hierarchy.dieu, hierarchy.khoan, hierarchy.diem
        ] if val
    ]
    hierarchy_str = " > ".join(hierarchy_parts) if hierarchy_parts else "Toàn văn"

    refs = extract_cross_references(chunk.text)
    refs_text = ""
    if refs:
        refs_formatted = [f"{r['target_type']} {r['target_number']} ({r['target_document']})" for r in refs[:3]]
        refs_text = "\nDẫn chiếu: " + "; ".join(refs_formatted)

    return f"""THÔNG TIN VĂN BẢN:
- Loại văn bản: {doc_type}
- Số hiệu: {doc_number}
- Ngày hiệu lực: {effective_date}
- Vị trí phân cấp: {hierarchy_str}{refs_text}

NỘI DUNG ĐOẠN TRÍCH:
{chunk.text[:1500]}
"""


build_llm_prompt = build_prompt


def build_kafka_producer() -> Producer:
    """Khởi tạo Kafka Producer phát đoạn trích lên Message Bus."""
    broker = getattr(config, "KAFKA_BROKER", "localhost:9092")
    conf = {
        "bootstrap.servers": broker,
        "acks": "all",
        "linger.ms": 20,
        "compression.type": "zstd",
        "batch.size": 65536,
    }
    return Producer(conf)


def process_document_record(record: Dict[str, Any], use_llm: bool = False, model: Optional[str] = None) -> List[Dict[str, Any]]:
    """Phân rã AST và làm giàu ngữ cảnh cho từng chunk của một văn bản."""
    doc_id = str(record.get("item_id") or record.get("doc_id") or record.get("id") or "")
    if not doc_id:
        return []

    html_raw = str(record.get("html_raw") or "")
    if not html_raw or record.get("html_status") != "VALID":
        return []

    import html2text
    from bs4 import BeautifulSoup
    
    soup = BeautifulSoup(html_raw, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "button", "iframe"]):
        tag.decompose()

    main_content = soup.find("div", class_="fulltext") or soup.find("body") or soup
    h2t = html2text.HTML2Text()
    h2t.ignore_links = True
    h2t.ignore_images = True
    h2t.body_width = 0
    raw_md = h2t.handle(str(main_content))

    meta_api = record.get("metadata_api") or {}
    meta_detail = record.get("metadata_detail") or {}

    doc_type = extract_doc_type(raw_md) or str((meta_api.get("docType") or {}).get("name") or "Văn bản")
    doc_number = extract_doc_number(raw_md) or str(record.get("doc_number") or doc_id)
    effective_date = (
        extract_effective_date(raw_md)
        or meta_detail.get("effFrom")
        or meta_api.get("effFrom")
        or "Chưa xác định"
    )

    cleaned_md = clean_boilerplate(raw_md)
    chunks = chunk_legal_document(cleaned_md, doc_id=doc_id, max_chunk_size=1600)

    output_chunks = []
    for chunk in chunks:
        if chunk.metadata.get("type") == "preamble" or len(chunk.text.strip()) < 50:
            continue

        if use_llm:
            prompt = build_prompt(chunk, doc_number, doc_type, effective_date)
            summary = call_llm(prompt, model=model)
            contextualized_text = f"{summary}\n\n{chunk.text}" if summary else build_deterministic_context(chunk, doc_number, doc_type, effective_date)
        else:
            contextualized_text = build_deterministic_context(chunk, doc_number, doc_type, effective_date)

        chunk_record = {
            "chunk_id": chunk.chunk_id,
            "doc_id": doc_id,
            "doc_number": doc_number,
            "original_text": chunk.text,
            "contextualized_text": contextualized_text,
            "metadata": {
                "doc_id": doc_id,
                "doc_number": doc_number,
                "doc_type": doc_type,
                "effective_date": str(effective_date),
                "source_doc": f"{doc_type} {doc_number}".strip(),
                "hierarchy_path": chunk.hierarchy.to_dict(),
                "chunk_level": chunk.level.value,
                "chunk_index": chunk.metadata.get("chunk_index", 1),
            },
        }
        output_chunks.append(chunk_record)

    return output_chunks


def contextualize_dataset(
    input_shards_dir: Path | str,
    output_path: Path | str,
    max_documents: Optional[int] = None,
    use_llm: bool = False,
    model: Optional[str] = None,
) -> int:
    """Quét toàn bộ tệp Shards trên đĩa và sản xuất contextual_chunks.jsonl."""
    in_dir = Path(input_shards_dir)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = out_file.with_suffix(f"{out_file.suffix}.tmp_{os.getpid()}")

    shard_files = sorted(list(in_dir.glob("crawl_pages_*.jsonl.gz")))
    if not shard_files:
        logger.error("Không tìm thấy tệp shard nào tại: %s", in_dir)
        return 0

    logger.info("Bắt đầu ngữ cảnh hóa: %d shards | Mode=%s", len(shard_files), "LLM" if use_llm else "Deterministic Structure")
    total_docs = 0
    total_chunks = 0

    with open(tmp_file, "w", encoding="utf-8") as out_f:
        for shard_path in shard_files:
            for record in read_jsonl(shard_path):
                if max_documents and total_docs >= max_documents:
                    break

                chunks = process_document_record(record, use_llm=use_llm, model=model)
                for ck in chunks:
                    out_f.write(json.dumps(ck, ensure_ascii=False) + "\n")
                    total_chunks += 1

                total_docs += 1
                if total_docs % 1000 == 0:
                    logger.info("Tiến độ: Đã xử lý %d văn bản -> %d chunks...", total_docs, total_chunks)

            if max_documents and total_docs >= max_documents:
                break

    tmp_file.replace(out_file)
    logger.info("✓ Hoàn tất xuất kho chunks ngữ cảnh: %d chunks từ %d văn bản -> %s", total_chunks, total_docs, out_file.resolve())
    return total_chunks


def main():
    parser = argparse.ArgumentParser(description="Điều phối tạo tập Contextual Chunks cho VietLawBERT")
    parser.add_argument("--input-dir", default=RAW_SHARDS_DIR, help="Thư mục chứa raw shards .jsonl.gz")
    parser.add_argument("--output", default=DATA_STORAGE_ROOT / "processed" / "contextual_chunks.jsonl", help="Tệp đầu ra")
    parser.add_argument("--max-docs", type=int, default=None, help="Giới hạn số văn bản cần xử lý (phục vụ test)")
    parser.add_argument("--use-llm", action="store_true", help="Bật LLM sinh tóm lược ngữ cảnh (chậm hơn)")
    parser.add_argument("--model", default=getattr(config, "CONTEXTUALIZER_MODEL", "qwen2.5:1.5b"))
    args = parser.parse_args()

    contextualize_dataset(
        input_shards_dir=args.input_dir,
        output_path=args.output,
        max_documents=args.max_docs,
        use_llm=args.use_llm,
        model=args.model,
    )


if __name__ == "__main__":
    main()
