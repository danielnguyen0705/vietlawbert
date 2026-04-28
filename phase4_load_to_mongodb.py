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


def make_stable_source_key(raw_doc: dict) -> str:
    """
    Dataset mới từ vbpl.vn search card chưa có URL chi tiết,
    nên cần fallback bằng id/document_id/title.
    """
    return (
        raw_doc.get("url")
        or raw_doc.get("source_url")
        or raw_doc.get("id")
        or raw_doc.get("document_id")
        or raw_doc.get("external_id")
        or raw_doc.get("title")
        or ""
    )


def make_document_key(doc: dict) -> str:
    base = (
        f"{doc.get('title', '')}|"
        f"{doc.get('document_number', '')}|"
        f"{doc.get('issuer', '')}|"
        f"{doc.get('source_url', '')}|"
        f"{doc.get('external_id', '')}"
    )
    return hashlib.md5(base.encode("utf-8")).hexdigest()


def upsert_source_links(db, links: list):
    inserted = 0

    for item in links:
        if isinstance(item, str):
            source_url = item
            source_site = "vbpl.vn"
            external_id = None
            title = None
        else:
            source_url = (
                item.get("url")
                or item.get("source_url")
                or item.get("document_id")
                or item.get("doc_id")
            )
            source_site = item.get("source_site") or item.get("domain") or "vbpl.vn"
            external_id = item.get("document_id")
            title = item.get("doc_id")

        if not source_url:
            continue

        record = {
            "source_url": source_url,
            "source_site": source_site,
            "external_id": external_id,
            "title": title,
            "crawl_status": "discovered",
            "updated_at": datetime.utcnow().isoformat(),
        }

        db.source_links.update_one(
            {"source_url": source_url},
            {"$set": record},
            upsert=True,
        )
        inserted += 1

    return inserted


def transform_document(raw_doc: dict) -> dict | None:
    source_url = make_stable_source_key(raw_doc)

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
        "external_id": raw_doc.get("id") or raw_doc.get("document_id"),
        "source_url": source_url,
        "original_url": raw_doc.get("url") or raw_doc.get("source_url") or "",
        "title": raw_doc.get("title"),
        "document_type": raw_doc.get("doc_type"),
        "document_number": raw_doc.get("doc_number"),
        "issuer": issuer,
        "issuer_codes": raw_doc.get("agency_codes", []),
        "issuers": issuer_list,
        "domain": raw_doc.get("domain"),
        "status": raw_doc.get("status"),
        "issued_date": raw_doc.get("issued_date"),
        "effective_date": raw_doc.get("effective_date"),
        "source_site": raw_doc.get("source_site"),
        "url_schema": raw_doc.get("url_schema"),
        "search_keyword": raw_doc.get("search_keyword"),
        "raw_text": raw_text,
        "cleaned_text": cleaned_text,
        "parse_status": raw_doc.get("parse_status", "success"),
        "created_at": raw_doc.get("created_at", datetime.utcnow().isoformat()),
        "updated_at": datetime.utcnow().isoformat(),
        "pipeline_version": "traffic-card-v1",
    }

    doc["document_key"] = make_document_key(doc)
    return doc


def upsert_documents(db, raw_documents: list):
    inserted = 0
    skipped = 0

    for raw_doc in raw_documents:
        doc = transform_document(raw_doc)

        if not doc:
            skipped += 1
            continue

        db.documents.update_one(
            {"source_url": doc["source_url"]},
            {"$set": doc},
            upsert=True,
        )
        inserted += 1

    return inserted, skipped


def build_document_id_map(db):
    mapping = {}

    for doc in db.documents.find({}, {"_id": 1, "source_url": 1, "external_id": 1, "title": 1}):
        if doc.get("source_url"):
            mapping[str(doc["source_url"])] = doc["_id"]

        if doc.get("external_id"):
            mapping[str(doc["external_id"])] = doc["_id"]

        if doc.get("title"):
            mapping[str(doc["title"])] = doc["_id"]

    return mapping


def find_chunk_document_id(raw_chunk: dict, document_id_map: dict):
    candidates = [
        raw_chunk.get("source_url"),
        raw_chunk.get("url"),
        raw_chunk.get("document_id"),
        raw_chunk.get("external_id"),
        raw_chunk.get("title"),
    ]

    for value in candidates:
        if value is None:
            continue

        key = str(value)
        if key in document_id_map:
            return document_id_map[key]

    return None


def transform_chunk(raw_chunk: dict, document_id_map: dict):
    document_id = find_chunk_document_id(raw_chunk, document_id_map)

    if not document_id:
        return None

    return {
        "document_id": document_id,
        "source_document_id": raw_chunk.get("document_id"),
        "chunk_id": raw_chunk.get("chunk_id"),
        "chunk_index": raw_chunk.get("chunk_index", 0),
        "section_index": raw_chunk.get("section_index"),
        "section_header": raw_chunk.get("section_header"),
        "chapter": raw_chunk.get("chapter"),
        "article": raw_chunk.get("article"),
        "clause": raw_chunk.get("clause"),
        "chunk_text": raw_chunk.get("text") or raw_chunk.get("chunk_text"),
        "char_count": raw_chunk.get("char_count"),
        "title": raw_chunk.get("title"),
        "domain": raw_chunk.get("domain"),
        "embedding_status": "pending",
        "created_at": datetime.utcnow().isoformat(),
    }


def upsert_chunks(db, raw_chunks: list, document_id_map: dict):
    inserted = 0
    skipped = 0

    for raw_chunk in raw_chunks:
        chunk = transform_chunk(raw_chunk, document_id_map)

        if not chunk:
            skipped += 1
            continue

        query = {
            "document_id": chunk["document_id"],
            "chunk_index": chunk["chunk_index"],
        }

        if chunk.get("chunk_id"):
            query = {"chunk_id": chunk["chunk_id"]}

        db.chunks.update_one(
            query,
            {"$set": chunk},
            upsert=True,
        )
        inserted += 1

    return inserted, skipped


def write_pipeline_log(db, status: str, document_count: int, chunk_count: int, message: str):
    db.pipeline_logs.insert_one(
        {
            "phase": "load_to_mongodb",
            "status": status,
            "document_count": document_count,
            "chunk_count": chunk_count,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )


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

    source_links_loaded = upsert_source_links(db, raw_links)

    documents_loaded, documents_skipped = upsert_documents(db, raw_documents)

    document_id_map = build_document_id_map(db)

    chunks_loaded, chunks_skipped = upsert_chunks(db, raw_chunks, document_id_map)

    write_pipeline_log(
        db=db,
        status="success",
        document_count=documents_loaded,
        chunk_count=chunks_loaded,
        message="Loaded data into MongoDB successfully.",
    )

    print("Load completed successfully.")
    print(f"Source links loaded: {source_links_loaded}")
    print(f"Documents loaded: {documents_loaded}")
    print(f"Documents skipped: {documents_skipped}")
    print(f"Chunks loaded: {chunks_loaded}")
    print(f"Chunks skipped: {chunks_skipped}")

    print("\nMongoDB counts:")
    print(f"source_links: {db.source_links.count_documents({})}")
    print(f"documents   : {db.documents.count_documents({})}")
    print(f"chunks      : {db.chunks.count_documents({})}")


if __name__ == "__main__":
    main()