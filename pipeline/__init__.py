"""
vietlawbert.pipeline
~~~~~~~~~~~~~~~~~~~~
Phân hệ điều phối xử lý dữ liệu trung gian ngoại tuyến (Decoupled ETL Worker):
Đọc Shards thô -> Bóc tách Hybrid AST -> Tiêm Metadata -> Mã hóa MRL (d=256) -> Nạp Qdrant & ES.
"""

from .ingest_pipeline import (
    IngestPipelineWorker,
    load_checkpoint,
    save_checkpoint,
)

__all__ = [
    "IngestPipelineWorker",
    "load_checkpoint",
    "save_checkpoint",
]