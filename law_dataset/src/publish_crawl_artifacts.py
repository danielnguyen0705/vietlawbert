"""Phát canonical crawl artifacts lên một Kafka topic mới, đúng một lần."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic

from audit_pilot import read_jsonl
from pipeline_artifacts import artifact_manifest, canonical_artifacts, encode_kafka_envelope
from run_crawl_shards import write_json_atomic


def topic_message_count(broker: str, topic: str) -> int:
    from confluent_kafka import Consumer, TopicPartition

    consumer = Consumer({"bootstrap.servers": broker, "group.id": "topic-preflight", "enable.auto.commit": False})
    try:
        metadata = consumer.list_topics(topic, timeout=10)
        if topic not in metadata.topics or metadata.topics[topic].error is not None:
            return 0
        return sum(
            consumer.get_watermark_offsets(TopicPartition(topic, partition), timeout=10)[1]
            for partition in metadata.topics[topic].partitions
        )
    finally:
        consumer.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish canonical crawl shards lên Kafka")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--expect-documents", type=int, required=True)
    parser.add_argument("--expect-shards", type=int, required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--partitions", type=int, default=8)
    parser.add_argument("--manifest-output", type=Path)
    args = parser.parse_args()
    broker = os.getenv("KAFKA_BROKER", "localhost:9092")

    paths = canonical_artifacts(args.input_dir.resolve(), args.expect_shards)
    manifest = artifact_manifest(paths)
    if len(manifest) != args.expect_documents:
        raise RuntimeError(f"expected {args.expect_documents} canonical IDs, got {len(manifest)}")

    admin = AdminClient({"bootstrap.servers": broker})
    result = admin.create_topics([NewTopic(args.topic, num_partitions=args.partitions, replication_factor=1)])
    try:
        result[args.topic].result(timeout=15)
    except Exception as exc:
        if "TOPIC_ALREADY_EXISTS" not in str(exc) and "already exists" not in str(exc).lower():
            raise
    existing = topic_message_count(broker, args.topic)
    if existing:
        raise RuntimeError(f"topic {args.topic} đã có {existing} message; dùng topic mới để gate chính xác")

    delivered = 0
    delivery_error = None

    def delivery_report(error, message):
        nonlocal delivered, delivery_error
        if error is not None:
            delivery_error = str(error)
        else:
            delivered += 1

    producer = Producer(
        {
            "bootstrap.servers": broker,
            "enable.idempotence": True,
            "acks": "all",
            "compression.type": "zstd",
            "linger.ms": 20,
            "batch.num.messages": 10000,
        }
    )
    for path in paths:
        for record in read_jsonl(path):
            payload = encode_kafka_envelope(record)
            if len(payload) >= 950_000:
                raise RuntimeError(
                    f"Kafka envelope vẫn quá lớn: {record['item_id']} = {len(payload)} bytes"
                )
            while True:
                try:
                    producer.produce(
                        args.topic,
                        key=str(record["item_id"]).encode("utf-8"),
                        value=payload,
                        callback=delivery_report,
                    )
                    break
                except BufferError:
                    producer.poll(0.5)
            producer.poll(0)
    remaining = producer.flush(120)
    if remaining or delivery_error or delivered != args.expect_documents:
        raise RuntimeError(
            f"Kafka delivery failed: delivered={delivered}, remaining={remaining}, error={delivery_error}"
        )

    output = args.manifest_output or (args.input_dir / "kafka_publish_manifest.json")
    write_json_atomic(
        output,
        {
            "topic": args.topic,
            "broker": broker,
            "documents": len(manifest),
            "delivered": delivered,
            "shards": [str(path) for path in paths],
        },
    )
    print(json.dumps({"topic": args.topic, "delivered": delivered}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
