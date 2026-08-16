"""Đối chiếu chính xác ID/hash giữa artifact, Kafka và Neo4j raw archive."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

from confluent_kafka import Consumer, TopicPartition
from neo4j import GraphDatabase

from artifacts.canonical import artifact_manifest, canonical_artifacts, decode_kafka_envelope, payload_hash
from crawler.shard_runner import write_json_atomic


def read_kafka_manifest(broker: str, topic: str):
    consumer = Consumer(
        {
            "bootstrap.servers": broker,
            "group.id": f"pipeline-audit-{uuid.uuid4()}",
            "enable.auto.commit": False,
        }
    )
    try:
        metadata = consumer.list_topics(topic, timeout=15)
        partitions = sorted(metadata.topics[topic].partitions)
        assignments = [TopicPartition(topic, partition, 0) for partition in partitions]
        high = {
            partition: consumer.get_watermark_offsets(TopicPartition(topic, partition), timeout=10)[1]
            for partition in partitions
        }
        consumer.assign(assignments)
        result = {}
        messages = 0
        while messages < sum(high.values()):
            message = consumer.poll(10)
            if message is None:
                raise RuntimeError("timeout khi audit Kafka")
            if message.error():
                raise RuntimeError(message.error())
            doc_id = message.key().decode("utf-8")
            if doc_id in result:
                raise ValueError(f"duplicate Kafka key: {doc_id}")
            result[doc_id] = payload_hash(decode_kafka_envelope(message.value()))
            messages += 1
        return result, high
    finally:
        consumer.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate artifact = Kafka = Neo4j cho một corpus")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--expect-shards", type=int, required=True)
    parser.add_argument("--expect-documents", type=int, required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    broker = os.getenv("KAFKA_BROKER", "localhost:9092")
    artifacts = artifact_manifest(canonical_artifacts(args.input_dir.resolve(), args.expect_shards))
    kafka, high = read_kafka_manifest(broker, args.topic)
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "vietlawbert")),
    )
    try:
        with driver.session() as session:
            database = {
                record["doc_id"]: record["payload_sha256"]
                for record in session.run(
                    "MATCH (d:RawLawDocument {dataset_id: $dataset_id}) "
                    "RETURN d.doc_id AS doc_id, d.payload_sha256 AS payload_sha256",
                    dataset_id=args.dataset_id,
                )
            }
            statuses = {
                record["status"]: record["total"]
                for record in session.run(
                    "MATCH (d:RawLawDocument {dataset_id: $dataset_id}) "
                    "RETURN d.html_status AS status, count(d) AS total",
                    dataset_id=args.dataset_id,
                )
            }
    finally:
        driver.close()
    result = {
        "artifact_documents": len(artifacts),
        "kafka_messages": len(kafka),
        "database_documents": len(database),
        "kafka_high_watermarks": high,
        "database_html_statuses": statuses,
        "artifact_kafka_id_match": artifacts.keys() == kafka.keys(),
        "artifact_database_id_match": artifacts.keys() == database.keys(),
        "artifact_kafka_hash_match": artifacts == kafka,
        "artifact_database_hash_match": artifacts == database,
    }
    result["passed"] = (
        len(artifacts) == len(kafka) == len(database) == args.expect_documents
        and all(value is True for key, value in result.items() if key.endswith("_match"))
    )
    write_json_atomic(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
