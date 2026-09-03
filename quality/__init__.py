"""
vietlawbert.quality
~~~~~~~~~~~~~~~~~~~
Phân hệ kiểm soát chất lượng và kiểm toán toàn vẹn dữ liệu cho VietLawBERT:
- Crawl Audit: Kiểm toán Shard nén Gzip, phân vùng cách ly và phát hiện rác dữ liệu.
- Database Inspection: Giám sát trạng thái hoạt động của Milvus, Neo4j và MongoDB.
- Pipeline Verifier: Đối soát toàn vẹn 3 chiều (Artifacts == Kafka == Neo4j Raw Archive).
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