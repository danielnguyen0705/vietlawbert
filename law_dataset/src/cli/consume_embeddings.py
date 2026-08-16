"""CLI chunk, embedding và nạp Milvus/Neo4j."""

import argparse
import os


def main() -> int:
    parser = argparse.ArgumentParser(description="VietLawBERT Kafka embedding consumer")
    parser.add_argument(
        "--idle-exit-seconds",
        type=float,
        default=float(os.getenv("CONSUMER_IDLE_EXIT_SECONDS", "0")),
        help="Thoát sau N giây Kafka không có message; 0 nghĩa là chạy daemon.",
    )
    args = parser.parse_args()

    # Import ML/database stack sau parse để `--help` vẫn dùng được trong môi
    # trường quản trị chưa cài PyTorch/Milvus client.
    from ingestion.embedding_consumer import LawEventConsumer

    consumer = LawEventConsumer()
    consumer.run(idle_exit_seconds=args.idle_exit_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
