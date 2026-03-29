from bson import ObjectId
from fastapi import FastAPI, Query
from app.db.mongo import get_database

app = FastAPI(title="VietLawBERT Data API")


def serialize_doc(doc):
    doc["_id"] = str(doc["_id"])
    return doc


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/db-check")
def db_check():
    db = get_database()
    return {"database": db.name}


@app.get("/documents")
def list_documents(
    document_type: str | None = Query(default=None),
    issuer: str | None = Query(default=None),
    limit: int = Query(default=20, le=100)
):
    db = get_database()
    query = {}

    if document_type:
        query["document_type"] = document_type
    if issuer:
        query["issuer"] = issuer

    docs = list(db.documents.find(query).limit(limit))
    return [serialize_doc(doc) for doc in docs]


@app.get("/documents/{document_id}")
def get_document(document_id: str):
    db = get_database()
    doc = db.documents.find_one({"_id": ObjectId(document_id)})
    if not doc:
        return {"error": "Document not found"}
    return serialize_doc(doc)


@app.get("/documents/{document_id}/chunks")
def get_document_chunks(document_id: str):
    db = get_database()
    chunks = list(
        db.chunks.find({"document_id": ObjectId(document_id)}).sort("chunk_index", 1)
    )
    for chunk in chunks:
        chunk["_id"] = str(chunk["_id"])
        chunk["document_id"] = str(chunk["document_id"])
    return chunks


@app.get("/search")
def search(keyword: str, limit: int = Query(default=10, le=50)):
    db = get_database()
    chunks = list(
        db.chunks.find(
            {"$text": {"$search": keyword}},
            {"score": {"$meta": "textScore"}}
        ).sort([("score", {"$meta": "textScore"})]).limit(limit)
    )

    for chunk in chunks:
        chunk["_id"] = str(chunk["_id"])
        chunk["document_id"] = str(chunk["document_id"])
    return chunks