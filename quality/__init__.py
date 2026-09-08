"""
vietlawbert.quality
~~~~~~~~~~~~~~~~~~~
Phân hệ kiểm soát chất lượng và kiểm toán toàn vẹn dữ liệu cho VietLawBERT (Kiến trúc v3):
- Crawl Audit: Kiểm toán Shard nén Gzip, tỷ lệ ký tự tiếng Việt và phát hiện phôi rác Template.
- Database Inspection: Giám sát trạng thái 5 tầng lưu trữ lai (Mongo, Neo4j, Qdrant, ES, Redis).
- Pipeline Verifier: Đối soát toàn vẹn 4 chiều (Disk Shards == Neo4j HIN == Qdrant == Elasticsearch).
"""

from .crawl_audit import audit_crawl, audit_databases
from .database_inspection import inspect_database, main as inspect_main
from .pipeline_verifier import verify_pipeline_lineage, main as verify_main

__all__ = [
    "audit_crawl",
    "audit_databases",
    "inspect_database",
    "inspect_main",
    "verify_pipeline_lineage",
    "verify_main",
]