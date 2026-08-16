"""CLI kiểm tra nhanh Milvus và Neo4j."""

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Kiểm tra nhanh Milvus và Neo4j")
    parser.parse_args()

    from quality.database_inspection import main as inspect

    return inspect()


if __name__ == "__main__":
    raise SystemExit(main())
