import json
import hashlib
from datetime import datetime
from pathlib import Path

from app.db.mongo import get_database

DATA_DIR = Path("law_dataset/data")
RAW_LINKS_FILE = DATA_DIR / "raw_links.json"
RAW_DOCUMENTS_FILE = DATA_DIR / "raw_documents.jsonl"
CLEANED_DOCUMENTS_FILE = DATA_DIR / "cleaned_documents.jsonl"
DOCUMENTS_FILE = DATA_DIR / "documents.jsonl"
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"


def normalize_path(path_str):
    if not path_str:
        return None
    return Path(str(path_str).replace("\\", "/"))


def read_text_file(path_str):
    path = normalize_path(path_str)
    if not path or not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="ignore")


def load_json(file_path: Path):
    if not file_path.exists():
        return []
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(file_path: Path):
    if not file_path.exists():
        return []
    rows = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def make_document_key(doc: dict) -> str:
    base = (
        f"{doc.get('title', '')}|"
        f"{doc.get('document_number', '')}|"
        f"{doc.get('issuer', '')}|"
        f"{doc.get('source_url', '')}"
    )
    return hashlib.md5(base.encode("utf-8")).hexdigest()


def upsert_source_links(db, links: list):
    for item in links:
        if isinstance(item, str):
            source_url = item
            source_site = "vbpl.vn"
        else:
            source_url = item.get("url") or item.get("source_url")
            source_site = item.get("domain", "vbpl.vn")

        if not source_url:
            continue

        record = {
            "source_url": source_url,
            "source_site": source_site,
            "crawl_status": "discovered",
            "updated_at": datetime.utcnow().isoformat()
        }

        db.source_links.update_one(
            {"source_url": source_url},
            {"$set": record},
            upsert=True
        )


def transform_document(raw_doc: dict) -> dict | None:
    source_url = raw_doc.get("url") or raw_doc.get("source_url")
    if not source_url:
        return None

    raw_text = raw_doc.get("raw_text")
    cleaned_text = raw_doc.get("cleaned_text")

    if not raw_text:
        raw_text = read_text_file(raw_doc.get("raw_text_path"))

    if not cleaned_text:
        cleaned_text = read_text_file(raw_doc.get("cleaned_text_path"))

    issuer_list = raw_doc.get("agencies") or []
    issuer = issuer_list[0] if issuer_list else None

    doc = {
        "external_id": raw_doc.get("id"),
        "source_url": source_url,
        "title": raw_doc.get("title"),
        "document_type": raw_doc.get("doc_type"),
        "document_number": raw_doc.get("doc_number"),
        "issuer": issuer,
        "issuer_codes": raw_doc.get("agency_codes", []),
        "issuers": issuer_list,
        "raw_text": raw_text,
        "cleaned_text": cleaned_text,
        "parse_status": "success",
        "created_at": raw_doc.get("created_at", datetime.utcnow().isoformat()),
        "updated_at": datetime.utcnow().isoformat(),
        "pipeline_version": "stage1-v1"
    }

    doc["document_key"] = make_document_key(doc)
    return doc


def upsert_documents(db, raw_documents: list):
    for raw_doc in raw_documents:
        doc = transform_document(raw_doc)
        if not doc:
            continue

        db.documents.update_one(
            {"source_url": doc["source_url"]},
            {"$set": doc},
            upsert=True
        )


def build_document_id_map(db):
    mapping = {}
    for doc in db.documents.find({}, {"_id": 1, "source_url": 1}):
        mapping[doc["source_url"]] = doc["_id"]
    return mapping


def transform_chunk(raw_chunk: dict, document_id_map: dict):
    source_url = raw_chunk.get("source_url") or raw_chunk.get("url")
    document_id = document_id_map.get(source_url)
    if not document_id:
        return None

    return {
        "document_id": document_id,
        "chunk_index": raw_chunk.get("chunk_index", 0),
        "section_header": raw_chunk.get("section_header"),
        "chapter": raw_chunk.get("chapter"),
        "article": raw_chunk.get("article"),
        "clause": raw_chunk.get("clause"),
        "chunk_text": raw_chunk.get("text") or raw_chunk.get("chunk_text"),
        "char_count": raw_chunk.get("char_count"),
        "embedding_status": "pending",
        "created_at": datetime.utcnow().isoformat()
    }


def upsert_chunks(db, raw_chunks: list, document_id_map: dict):
    for raw_chunk in raw_chunks:
        chunk = transform_chunk(raw_chunk, document_id_map)
        if not chunk:
            continue

        db.chunks.update_one(
            {
                "document_id": chunk["document_id"],
                "chunk_index": chunk["chunk_index"]
            },
            {"$set": chunk},
            upsert=True
        )


def write_pipeline_log(db, status: str, document_count: int, chunk_count: int, message: str):
    db.pipeline_logs.insert_one({
        "phase": "load_to_mongodb",
        "status": status,
        "document_count": document_count,
        "chunk_count": chunk_count,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    })


def main():
    db = get_database()

    raw_links = load_json(RAW_LINKS_FILE)

    raw_documents = load_jsonl(CLEANED_DOCUMENTS_FILE)
    if not raw_documents:
        raw_documents = load_jsonl(RAW_DOCUMENTS_FILE)
    if not raw_documents:
        raw_documents = load_jsonl(DOCUMENTS_FILE)

    raw_chunks = load_jsonl(CHUNKS_FILE)

    print(f"Raw links found: {len(raw_links)}")
    print(f"Documents found: {len(raw_documents)}")
    print(f"Chunks found: {len(raw_chunks)}")

    upsert_source_links(db, raw_links)
    upsert_documents(db, raw_documents)

    document_id_map = build_document_id_map(db)
    upsert_chunks(db, raw_chunks, document_id_map)

    write_pipeline_log(
        db=db,
        status="success",
        document_count=len(raw_documents),
        chunk_count=len(raw_chunks),
        message="Loaded data into MongoDB successfully."
    )

    print("Load completed successfully.")
    print(f"Documents loaded: {len(raw_documents)}")
    print(f"Chunks loaded: {len(raw_chunks)}")


if __name__ == "__main__":
    main()