"""
vietlawbert.pipeline
~~~~~~~~~~~~~~~~~~~~
Phân hệ điều phối xử lý dữ liệu trung gian ngoại tuyến (Decoupled ETL Worker):
- ingest_pipeline: Đọc Shards thô -> Bóc tách Hybrid AST -> Mã hóa MRL (d=256) -> Bulk nạp đồng thời Qdrant & ES.
"""

from .ingest_pipeline import IngestPipelineWorker

__all__ = ["IngestPipelineWorker"]