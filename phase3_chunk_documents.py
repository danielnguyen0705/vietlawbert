from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


BASE_DIR = Path(__file__).resolve().parent
LAW_DATASET_DIR = BASE_DIR / "law_dataset"
DATA_DIR = LAW_DATASET_DIR / "data"

DOCUMENTS_JSONL = DATA_DIR / "documents.jsonl"
CHUNKS_JSONL = DATA_DIR / "chunks.jsonl"
CHUNK_TEXT_DIR = DATA_DIR / "chunks"
CHUNK_LOGS_DIR = DATA_DIR / "logs"
CHUNK_ERRORS_JSONL = CHUNK_LOGS_DIR / "chunk_errors.jsonl"

DEFAULT_MAX_CHARS = 1200
DEFAULT_OVERLAP_CHARS = 150
MIN_CHUNK_CHARS = 120

HEADER_PATTERNS = [
    r"^CHƯƠNG\s+[IVXLCDM0-9]+",
    r"^MỤC\s+[IVXLCDM0-9]+",
    r"^PHẦN\s+[IVXLCDM0-9]+",
    r"^ĐIỀU\s+\d+[\.:]?",
    r"^KHOẢN\s+\d+[\.:]?",
    r"^\d+\.\s+",
    r"^\([0-9]+\)",
]


@dataclass
class ChunkConfig:
    max_chars: int = DEFAULT_MAX_CHARS
    overlap_chars: int = DEFAULT_OVERLAP_CHARS
    min_chunk_chars: int = MIN_CHUNK_CHARS


# ---------- basic io ----------
def ensure_dirs() -> None:
    CHUNK_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    CHUNK_LOGS_DIR.mkdir(parents=True, exist_ok=True)



def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                append_jsonl(
                    CHUNK_ERRORS_JSONL,
                    {
                        "line_no": line_no,
                        "path": str(path),
                        "error": f"JSONDecodeError: {e}",
                    },
                )



def append_jsonl(path: Path, record: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")



def write_text_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ---------- normalize ----------
def norm_text(text: str) -> str:
    text = (text or "").replace("\xa0", " ").replace("\u200b", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()



def is_header_line(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    return any(re.match(pat, s, flags=re.IGNORECASE) for pat in HEADER_PATTERNS)



def split_into_sections(text: str) -> List[Dict]:
    """
    Tách thô theo các line heading như CHƯƠNG / MỤC / ĐIỀU.
    Giữ heading đi kèm nội dung section để chunk sau này dễ trace hơn.
    """
    text = norm_text(text)
    if not text:
        return []

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    sections: List[Dict] = []

    current_header = ""
    current_lines: List[str] = []

    def flush_section() -> None:
        nonlocal current_header, current_lines
        body = "\n".join(current_lines).strip()
        if current_header or body:
            sections.append(
                {
                    "header": current_header.strip(),
                    "text": norm_text((current_header + "\n" + body).strip()) if current_header else norm_text(body),
                }
            )
        current_header = ""
        current_lines = []

    for line in lines:
        if is_header_line(line):
            if current_header or current_lines:
                flush_section()
            current_header = line
        else:
            current_lines.append(line)

    if current_header or current_lines:
        flush_section()

    if not sections:
        sections.append({"header": "", "text": text})

    return sections



def chunk_long_text(text: str, config: ChunkConfig) -> List[str]:
    text = norm_text(text)
    if not text:
        return []

    if len(text) <= config.max_chars:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks: List[str] = []
    current = ""

    def push_current() -> None:
        nonlocal current
        if current.strip():
            chunks.append(norm_text(current))
            current = ""

    for para in paragraphs:
        candidate = para if not current else current + "\n\n" + para
        if len(candidate) <= config.max_chars:
            current = candidate
            continue

        if current:
            push_current()

        if len(para) <= config.max_chars:
            current = para
            continue

        start = 0
        while start < len(para):
            end = min(start + config.max_chars, len(para))
            piece = para[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= len(para):
                break
            start = max(end - config.overlap_chars, start + 1)

    if current:
        push_current()

    merged: List[str] = []
    for chunk in chunks:
        if merged and len(chunk) < config.min_chunk_chars:
            candidate = merged[-1] + "\n\n" + chunk
            if len(candidate) <= config.max_chars + config.overlap_chars:
                merged[-1] = norm_text(candidate)
                continue
        merged.append(norm_text(chunk))

    return merged



def build_chunks_for_document(doc: Dict, config: ChunkConfig) -> List[Dict]:
    cleaned_text_path = doc.get("cleaned_text_path") or ""
    if not cleaned_text_path:
        raise ValueError("Missing cleaned_text_path")

    text_path = BASE_DIR / cleaned_text_path if not Path(cleaned_text_path).is_absolute() else Path(cleaned_text_path)
    if not text_path.exists():
        raise FileNotFoundError(f"cleaned text file not found: {text_path}")

    cleaned_text = text_path.read_text(encoding="utf-8")
    cleaned_text = norm_text(cleaned_text)
    if not cleaned_text:
        return []

    sections = split_into_sections(cleaned_text)
    chunks: List[Dict] = []
    chunk_index = 0

    for section_index, section in enumerate(sections, start=1):
        section_text = section["text"].strip()
        if not section_text:
            continue

        section_chunks = chunk_long_text(section_text, config)
        for local_idx, chunk_text in enumerate(section_chunks, start=1):
            chunk_index += 1
            chunk_id = f'{doc.get("id") or "unknown"}_chunk_{chunk_index:04d}'
            chunk_rel_path = Path("law_dataset/data/chunks") / f"{chunk_id}.txt"
            write_text_file(BASE_DIR / chunk_rel_path, chunk_text)

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": doc.get("id", ""),
                    "doc_number": doc.get("doc_number", ""),
                    "doc_type": doc.get("doc_type", ""),
                    "title": doc.get("title", ""),
                    "ministry": doc.get("ministry", ""),
                    "agency_codes": doc.get("agency_codes", []),
                    "agencies": doc.get("agencies", []),
                    "url": doc.get("url", ""),
                    "domain": doc.get("domain", doc.get("type", "")),
                    "section_index": section_index,
                    "section_header": section.get("header", ""),
                    "chunk_index": chunk_index,
                    "chunk_index_in_section": local_idx,
                    "chunk_text_path": str(chunk_rel_path).replace("\\", "/"),
                    "char_count": len(chunk_text),
                    "text": chunk_text,
                }
            )

    return chunks



def main() -> None:
    ensure_dirs()

    if not DOCUMENTS_JSONL.exists():
        raise FileNotFoundError(f"documents.jsonl not found: {DOCUMENTS_JSONL}")

    if CHUNKS_JSONL.exists():
        CHUNKS_JSONL.unlink()
    if CHUNK_ERRORS_JSONL.exists():
        CHUNK_ERRORS_JSONL.unlink()

    config = ChunkConfig()

    total_docs = 0
    total_chunks = 0
    ok_docs = 0
    failed_docs = 0

    for doc in iter_jsonl(DOCUMENTS_JSONL):
        total_docs += 1
        try:
            chunks = build_chunks_for_document(doc, config)
            for chunk in chunks:
                append_jsonl(CHUNKS_JSONL, chunk)
            total_chunks += len(chunks)
            ok_docs += 1
            print(f"[OK] {doc.get('id', '')} -> {len(chunks)} chunks")
        except Exception as e:
            failed_docs += 1
            append_jsonl(
                CHUNK_ERRORS_JSONL,
                {
                    "document_id": doc.get("id", ""),
                    "doc_number": doc.get("doc_number", ""),
                    "title": doc.get("title", ""),
                    "cleaned_text_path": doc.get("cleaned_text_path", ""),
                    "error": f"{type(e).__name__}: {e}",
                },
            )
            print(f"[ERROR] {doc.get('id', '')}: {e}")

    print("\n=== CHUNK SUMMARY ===")
    print(f"Documents processed: {total_docs}")
    print(f"Documents succeeded: {ok_docs}")
    print(f"Documents failed   : {failed_docs}")
    print(f"Total chunks       : {total_chunks}")
    print(f"Output JSONL       : {CHUNKS_JSONL}")


if __name__ == "__main__":
    main()