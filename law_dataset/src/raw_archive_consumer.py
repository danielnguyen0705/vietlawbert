"""Consumer nhanh: Kafka → Neo4j raw-document manifest, không embedding/model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

from confluent_kafka import Consumer
from neo4j import GraphDatabase

from artifacts.canonical import decode_kafka_envelope


def write_batch(session, rows):
    session.run(
        """
        UNWIND $rows AS row
        MERGE (d:RawLawDocument {dataset_id: row.dataset_id, doc_id: row.doc_id})
        SET d.payload_sha256 = row.payload_sha256,
            d.payload_bytes = row.payload_bytes,
            d.html_status = row.html_status,
            d.content_source = row.content_source,
            d.rescue_status = row.rescue_status,
            d.doc_number = row.doc_number,
            d.title = row.title,
            d.archived_at = datetime()
        """,
        rows=rows,
    ).consume()


def main() -> int:
    parser = argparse.ArgumentParser(description="Archive Kafka raw docs vào Neo4j không embedding")
    parser.add_argument("--topic", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--expect-documents", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--idle-exit-seconds", type=float, default=10)
    args = parser.parse_args()
    broker = os.getenv("KAFKA_BROKER", "localhost:9092")
    group = os.getenv("KAFKA_RAW_GROUP_ID", f"raw-archive-{args.dataset_id}")
    consumer = Consumer(
        {
            "bootstrap.servers": broker,
            "group.id": group,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
            "max.poll.interval.ms": 900000,
        }
    )
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "vietlawbert")),
    )
    started = time.monotonic()
    last_message = started
    consumed = 0
    batch = []
    try:
        with driver.session() as session:
            session.run(
                "CREATE CONSTRAINT raw_law_document_id IF NOT EXISTS "
                "FOR (d:RawLawDocument) REQUIRE (d.dataset_id, d.doc_id) IS UNIQUE"
            ).consume()
            consumer.subscribe([args.topic])
            while True:
                message = consumer.poll(1.0)
                if message is None:
                    if time.monotonic() - last_message >= args.idle_exit_seconds:
                        break
                    continue
                if message.error():
                    raise RuntimeError(message.error())
                payload = decode_kafka_envelope(message.value())
                record = json.loads(payload)
                detail = record.get("metadata_detail") or {}
                batch.append(
                    {
                        "dataset_id": args.dataset_id,
                        "doc_id": str(record["item_id"]),
                        "payload_sha256": hashlib.sha256(payload).hexdigest(),
                        "payload_bytes": len(payload),
                        "html_status": str(record.get("html_status") or ""),
                        "content_source": str(record.get("content_source") or ""),
                        "rescue_status": str(record.get("rescue_status") or ""),
                        "doc_number": str(record.get("doc_number") or ""),
                        "title": str(detail.get("title") or ""),
                    }
                )
                consumed += 1
                last_message = time.monotonic()
                if len(batch) >= args.batch_size:
                    write_batch(session, batch)
                    consumer.commit(asynchronous=False)
                    batch.clear()
            if batch:
                write_batch(session, batch)
                consumer.commit(asynchronous=False)
                batch.clear()
            stored = session.run(
                "MATCH (d:RawLawDocument {dataset_id: $dataset_id}) RETURN count(d) AS total",
                dataset_id=args.dataset_id,
            ).single()["total"]
    finally:
        consumer.close()
        driver.close()
    elapsed = time.monotonic() - started
    result = {
        "topic": args.topic,
        "group": group,
        "consumed": consumed,
        "stored": stored,
        "elapsed_seconds": round(elapsed, 3),
        "documents_per_second": round(consumed / elapsed, 3) if elapsed else 0,
        "passed": stored == args.expect_documents,
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
