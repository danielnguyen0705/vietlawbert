from app.db.mongo import get_database


def create_indexes():
    db = get_database()

    db.documents.create_index("source_url", unique=True)
    db.documents.create_index("document_number")
    db.documents.create_index("document_type")
    db.documents.create_index("issuer")
    db.documents.create_index([("title", "text"), ("cleaned_text", "text")])

    db.chunks.create_index("document_id")
    db.chunks.create_index("chunk_index")
    db.chunks.create_index([("chunk_text", "text")])

    db.pipeline_logs.create_index("phase")
    db.pipeline_logs.create_index("status")
    db.pipeline_logs.create_index("timestamp")


if __name__ == "__main__":
    create_indexes()
    print("Indexes created successfully.")